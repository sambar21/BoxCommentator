# Box.IO AI Commentator

**Real-time AI boxing commentary engine, punch by punch play by play. Latency and quality numbers are being measured across self-hosted vLLM and hosted APIs; see [Status](#status).**

Feed it raw fight data. Six statistical trackers build a living model of the fight. A priority queue with exponential time decay surfaces the most interesting events. A dual track LLM generator produces commentary, analytical breakdowns for evolving patterns, instant reactions for explosive moments, grounded by a pgvector RAG pipeline that retrieves live fighter stats and historical match data. A Go API gateway runs LLM inference and event telemetry concurrently and cuts off requests that exceed a 550ms budget.


## Status

Turning this into a measured LLM-serving project: self-hosted Qwen2.5 on vLLM vs Nebius Token Factory vs Groq. Full plan: [`docs/BUILD_PLAN.md`](docs/BUILD_PLAN.md). Run guide: [`docs/RUNNING.md`](docs/RUNNING.md).

**Built (all unit-tested; no real benchmark has been run yet)**
- Knockdown events and a seeded 30-fight simulator with ground-truth knockdowns (`bench/fights.py`)
- Benchmark runner with spend/request caps and rate pacing (`bench/bench.py`), knockdown-recall and invented-stat checks (`bench/quality.py`), charts and results table (`bench/charts.py`)
- Per-generation timing (time to first token, total, tokens, cache hits) and prompts laid out for prefix caching
- OpenAI-compatible backends config (vLLM, Nebius, Groq) and per-track routing with fallback
- vLLM serving variants and experiment plan (`serving/`) and a Kaggle notebook
- LoRA data generation (teacher distillation) and training (`finetune/`), Kokoro voice with time-to-first-audio (`voice/`), kind manifests (`k8s/`)

**Results:** none yet. Nothing in this README claims a measured number until it appears in `bench/results/`.

## Architecture

```
                         ┌─────────────────────────────────┐
  POST /api/v1/punch     │         Go API Gateway          │
  (fight event) ────────>│            :8080                │
                         │                                  │
                         │  punch_handler.go                │
                         │        │                         │
                         │   dispatcher.go (Goroutines)     │
                         │   ┌────┴────────┐               │
                         │   │             │               │
                         │ goroutine     goroutine         │
                         │ commentary    telemetry         │
                         │ (550ms max)   (fire & forget)   │
                         │   │                             │
                         │ metrics.go (p95 histogram)      │
                         └───┼─────────────────────────────┘
                             │ HTTP POST /internal/punch
                             ▼
                    ┌──────────────────────────┐
                    │   Python FastAPI Service  │
                    │          :8000            │
                    │                           │
                    │  CommentaryOrchestrator   │
                    │         │                 │
                    │    ┌────┴────┐            │
                    │  Buffer  Trackers (6x)    │
                    │    └────┬────┘            │
                    │      Events               │
                    │         │                 │
                    │   PriorityQueue           │
                    │  (score=p×2^(-age/5s))    │
                    │         │                 │
                    │  QueueConsumer (60/40)    │
                    │   ┌─────┴──────┐         │
                    │ Track A     Track B       │
                    │ (streaming) (urgent)      │
                    │   └─────┬──────┘         │
                    │  ContextBuilder           │
                    │         │                 │
                    │   RAG Retriever ─────────>│──> pgvector
                    │  (LangChain chain)        │    (PostgreSQL)
                    │         │                 │
                    │    Claude / Groq          │
                    │         │                 │
                    │    Commentary             │
                    └──────────────────────────┘
```

### Request flow (one punch event):

```
1. Client → Go gateway  POST /api/v1/punch  {attacker, punch_type, target, outcome}
2. Go spawns goroutine A → POST Python FastAPI /internal/punch
3. Go spawns goroutine B → logs telemetry (timestamp, fight_id, event_type)
4. Python: adds punch to ActionBuffer
5. Python: 6 trackers independently update → emit typed Events
6. Python: ContradictionDetector cross-checks tracker states
7. Python: CooldownManager deduplicates per-type
8. Python: PriorityQueue scores each event: score = priority × 2^(-age / 5s)
9. Python: QueueConsumer routes to Track A or Track B
10. Python: ContextBuilder assembles LLM prompt + RAG-retrieved fighter stats
11. Python: LLM generates commentary (streaming for Track A, direct for Track B)
12. Go goroutine A returns commentary → gateway responds to client
13. Go metrics.go records latency; requests slower than 550ms are cut off with a 504
```

## What's Built

### Core Pipeline (Python)

| Component | File | What it does |
|---|---|---|
| ActionBuffer | `src/core/action_buffer/buffer.py` | Circular `deque` (last 20 punches, O(1) append) |
| Event | `src/core/events/events.py` | Typed event dataclass, type, priority, message, context |
| PriorityQueue | `src/core/priority_queue/hot_queue.py` | Scored queue, `score = priority × 2^(-age/5s)`, staleness pruning |
| CooldownManager | `src/core/cooldown/manager.py` | Per type cooldown timers, priority override bypass |
| QueueConsumer | `src/core/queue_consumer/consumer.py` | 60/40 Track A/B balance routing |
| ContextBuilder | `src/core/context_builder/builder.py` | Assembles LLM prompt from tracker states + recent actions |
| Orchestrator | `src/core/orchestrator.py` | Central coordinator, punches in, commentary out |

### Six Trackers

| Tracker | File | Algorithm |
|---|---|---|
| Dominance | `src/trackers/dominance/tracker.py` | Pendulum state machine over a doubly linked list of 5 nodes (`P2_DOM ↔ P2_EDGE ↔ EVEN ↔ P1_EDGE ↔ P1_DOM`) |
| Pace | `src/trackers/pace/tracker.py` | Punch frequency with hysteresis (separate entry/exit thresholds, prevents oscillation) |
| Momentum | `src/trackers/momentum/tracker.py` | Sliding window comparison: recent 10 punches vs prior 10 punches |
| Excitement | `src/trackers/excitement/tracker.py` | Action density + idle timeout detection (4s lull → filler commentary) |
| Targets | `src/trackers/targets/tracker.py` | Head/body landing ratio to detect strategic shifts |
| Round Context | `src/trackers/round_context/tracker.py` | Early/mid/late narrative framing per round phase |

### Dual Track LLM Generation

```
Track A (Analytical)                    Track B (Urgent)
────────────────────                    ────────────────
For: strategy, patterns, technique      For: knockdowns, big combos, hurt fighters
Pace: conversational                    Pace: explosive, immediate
Generation: streaming (300ms)           Generation: direct (200ms)
Interruptible: yes                      Interruptible: no
Temperature: 0.7                        Temperature: 0.9
Max tokens: 40                          Max tokens: 50

Interrupt protocol:
  If Track A is streaming and a Track B event (priority ≥ 9.0) fires →
  TrackAGenerator.interrupt() sets flag → generator yields stop → Track B takes over
```

### LLM Provider Abstraction

```python
PROVIDER = "groq"   # → GroqClient (free, ~200ms, llama-3.3-70b)
PROVIDER = "claude" # → ClaudeClient (~500ms, claude-sonnet-4-5)
PROVIDER = "openai" # → OpenAIClient (~400ms, gpt-4o-mini)
PROVIDER = "ollama" # → OllamaClient (free, local, variable)
```

All behind `BaseLLMClient`, swap one line in `src/config/llm_config.py`, zero code changes.

## What's Being Built

### Phase 1, RAG Pipeline (pgvector + LangChain)

**Problem:** Commentary is ungrounded, the LLM hallucinates stats ("Garcia has a 70% KO rate!", wrong). Need real fighter data retrieved at inference time.

**Solution:** pgvector stores fighter profiles + historical fight summaries as embeddings. At fight start, seed fighter data. On every commentary request, retrieve top-k relevant stats and inject into prompt.

```
src/retrieval/
├── stats/
│   ├── fighter_stats.py          # FighterStats dataclass: name, record, style, KO%, reach, stance
│   └── seeder.py                 # Seeds sample fighters + fight histories into pgvector
├── fight_memory/
│   └── store.py                  # LangChain PGVector store — upsert + retrieve fight memory
├── historical_search/
│   └── searcher.py               # Semantic search: embed query → cosine similarity → top-k results
└── sliding_window/
    └── analyzer.py               # Live stats computed from ActionBuffer (landing%, combo rate)
```

**LangChain RAG chain:**
```python
# At fight start:
seeder.seed_fighters("Alvarez", "Garcia")  # embeds + stores fighter profiles

# On every commentary request:
retriever = store.as_retriever(search_kwargs={"k": 3})
relevant_stats = retriever.get_relevant_documents(query=event_focus)
# → injects "Alvarez: 60-1, orthodox, 78% KO rate, 70.5in reach" into prompt
```

**Why pgvector over a regular DB:**

Fighter profiles are unstructured text, "aggressive pressure fighter with a high guard, tends to walk opponents down"
Semantic search finds *conceptually relevant* stats, not just exact matches
Real-time query: "who lands body shots effectively?" → embedding → nearest neighbors → Canelo stats returned

### Phase 2, Python FastAPI Service

Wraps the orchestrator in an HTTP API so the Go gateway can call it.

```
src/api/
├── server.py       # FastAPI app, uvicorn runner
└── routes.py       # POST /internal/punch → orchestrator → {commentary, latency_ms}
                    # GET  /health         → {"status": "ok", "provider": "groq"}
```

**Punch endpoint:**
```python
@app.post("/internal/punch")
async def process_punch(punch: PunchRequest) -> CommentaryResponse:
    t0 = time.perf_counter()
    commentary = orchestrator.process_punch(punch.to_domain())
    return CommentaryResponse(
        commentary=commentary,
        latency_ms=(time.perf_counter() - t0) * 1000
    )
```

### Phase 3, Go API Gateway

High-concurrency HTTP server. Main job: parallelize LLM inference with event telemetry so neither blocks the other.

```
gateway/
├── main.go                     # HTTP server :8080, router, graceful shutdown
├── go.mod
├── handlers/
│   └── punch_handler.go        # POST /api/v1/punch, validates, dispatches
├── parallel/
│   └── dispatcher.go           # Fan-out goroutines: commentary + telemetry
├── telemetry/
│   └── metrics.go              # Rolling p95 latency histogram, event counters
└── client/
    └── python_client.go        # HTTP client to Python FastAPI service
```

**The goroutine fan-out (this is the core latency win):**
```go
// dispatcher.go
func Dispatch(punch PunchEvent) (string, error) {
    commentaryCh := make(chan CommentaryResult, 1)
    
    // Goroutine 1: call Python service for commentary
    go func() {
        result, err := pythonClient.PostPunch(punch)
        commentaryCh <- CommentaryResult{result, err}
    }()
    
    // Goroutine 2: fire-and-forget telemetry (doesn't block response)
    go func() {
        telemetry.Record(punch, time.Now())
    }()
    
    // Wait on commentary with hard timeout
    select {
    case result := <-commentaryCh:
        return result.Commentary, result.Err
    case <-time.After(550 * time.Millisecond):
        return "", ErrTimeout  // bounds worst-case wait at 550ms
    }
}
```

**What the fan-out buys (illustrative, not yet measured):**

Without goroutines: commentary call (400ms) + telemetry logging (80ms) = 480ms sequential
With goroutines: max(400ms commentary, 80ms telemetry) = 400ms parallel
The telemetry goroutine is a log write, so the saving is small; the timeout is the more important guarantee. Real numbers will come from `bench/` and replace the example figures above.

**p95 latency tracking:**
```go
// metrics.go — rolling histogram of last 1000 requests
func (m *Metrics) Record(latencyMs float64) {
    m.mu.Lock()
    m.samples = append(m.samples, latencyMs)
    if len(m.samples) > 1000 { m.samples = m.samples[1:] }
    m.mu.Unlock()
}

func (m *Metrics) P95() float64 {
    sorted := sorted(m.samples)
    return sorted[int(float64(len(sorted))*0.95)]
}
```

### Phase 4, Docker Compose

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: boxio
      POSTGRES_PASSWORD: boxio
    volumes:
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql

  python-service:
    build: .
    command: uvicorn src.api.server:app --host 0.0.0.0 --port 8000
    depends_on: [postgres]
    environment:
      DATABASE_URL: postgresql://postgres:boxio@postgres:5432/boxio
      GROQ_API_KEY: ${GROQ_API_KEY}

  go-gateway:
    build: ./gateway
    ports: ["8080:8080"]
    depends_on: [python-service]
    environment:
      PYTHON_SERVICE_URL: http://python-service:8000
```

One command to run the full stack: `docker compose up`

## Design Decisions

| Problem | Solution | Why |
|---|---|---|
| Dominance isn't binary, fighters drift in and out of control | Pendulum state machine (5 node doubly linked list) | Enforces ordered transitions; can't jump "even" → "dominating" without passing "edge" |
| Pace tracker oscillates near thresholds | Hysteresis, different entry/exit thresholds per state | Standard signal processing technique; eliminates flip flopping at boundary values |
| Old events clog the queue and produce stale commentary | Exponential half life decay `2^(-age/5s)` | A 10s old event scores at 25% of a fresh one; freshness always wins |
| Commentator repeats itself every 2 seconds | Per type cooldown timers with priority override | Blocks repeats; but a 9.5+ priority event (knockdown) always bypasses, because knockdowns always get called |
| Need both thoughtful analysis and instant reactions | Dual track generation with interrupt protocol | Track A streams for 300ms; if Track B event fires, cancels mid stream and takes over |
| LLM hallucinates fighter stats during live streams | pgvector RAG, real stats retrieved at inference time | Grounds every commentary line to real data; retrieval is semantic so "aggressive style" query returns stylistically relevant fighters |
| Tracker states can contradict each other | ContradictionDetector synthesizes tension events | "P1 dominating but P2 building momentum" becomes its own high priority narrative event |
| Switching LLM providers shouldn't break anything | Abstract `BaseLLMClient` + factory pattern | Swap `PROVIDER = "groq"` to `"claude"`, zero code changes |
| LLM goes down mid fight | Template based `EventSynthesizer` fallback | Graceful degradation, commentary quality drops but system never crashes |
| Go gateway and Python LLM call can't both block response time | Goroutine fan out with 550ms hard timeout | Telemetry never waits on the LLM call; the timeout bounds the worst-case wait (measured p95 will be reported in Status) |

## Project Structure

```
.
├── mock_fight.py                          # Python-only demo — simulates 60s round
├── requirements.txt
├── docker-compose.yml                     # Full stack: postgres + python + go
├── Dockerfile                             # Python service image
│
├── gateway/                               # Go API Gateway
│   ├── main.go                            # HTTP server :8080
│   ├── go.mod
│   ├── handlers/
│   │   └── punch_handler.go               # POST /api/v1/punch
│   ├── parallel/
│   │   └── dispatcher.go                  # Goroutine fan-out (commentary + telemetry)
│   ├── telemetry/
│   │   └── metrics.go                     # p95 latency histogram
│   └── client/
│       └── python_client.go               # HTTP client to Python service
│
├── scripts/
│   └── init.sql                           # CREATE EXTENSION vector; schema setup
│
└── src/
    ├── api/                               # Python FastAPI service
    │   ├── server.py                      # FastAPI app + uvicorn
    │   └── routes.py                      # POST /internal/punch, GET /health
    │
    ├── config/
    │   ├── llm_config.py                  # Provider selection + model params
    │   └── tts_config.py                  # ElevenLabs voice settings
    │
    ├── core/
    │   ├── orchestrator.py                # Central pipeline coordinator
    │   ├── action_buffer/buffer.py        # Circular deque (last 20 punches)
    │   ├── context_builder/builder.py     # Assembles LLM prompt from state + RAG
    │   ├── cooldown/manager.py            # Per-type cooldown with override
    │   ├── events/events.py               # Event dataclass
    │   ├── priority_queue/hot_queue.py    # Scored queue with decay + staleness pruning
    │   └── queue_consumer/consumer.py     # Track A/B routing + balance
    │
    ├── generation/
    │   ├── llm_interface/                 # BaseLLMClient, 4 providers, factory
    │   ├── speech_synthesis/              # TTS engine + pygame audio pipeline
    │   ├── track_a/generator.py           # Analytical (streaming, interruptible)
    │   └── track_b/generator.py           # Urgent (direct, fast)
    │
    ├── retrieval/                         # RAG Pipeline (pgvector + LangChain)
    │   ├── stats/
    │   │   ├── fighter_stats.py           # FighterStats dataclass
    │   │   └── seeder.py                  # Seeds fighter data + embeddings
    │   ├── fight_memory/
    │   │   └── store.py                   # LangChain PGVector store
    │   ├── historical_search/
    │   │   └── searcher.py                # Semantic search over fight history
    │   └── sliding_window/
    │       └── analyzer.py                # Live per-round stats from ActionBuffer
    │
    ├── synthesis/
    │   ├── aggregator/event_synthesizer.py  # Template fallback when LLM unavailable
    │   └── contradiction_detector/          # Cross-tracker narrative tension
    │
    └── trackers/
        ├── dominance/                     # Pendulum state machine
        ├── excitement/                    # Action density + idle timeout
        ├── momentum/                      # Sliding-window comparison
        ├── pace/                          # Tempo with hysteresis
        ├── round_context/                 # Phase-aware narrative framing
        └── targets/                       # Head/body targeting patterns
```

## Quick Start

### Python only (no Docker required):

```bash
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY (free at console.groq.com)
python mock_fight.py
```

### Full stack (Go gateway + RAG):

```bash
cp .env.example .env
# Fill in GROQ_API_KEY and optionally ANTHROPIC_API_KEY

docker compose up
# Starts PostgreSQL+pgvector, Python FastAPI service, Go gateway

# In another terminal — send a punch event:
curl -X POST http://localhost:8080/api/v1/punch \
  -H "Content-Type: application/json" \
  -d '{"attacker":1,"punch_type":"hook","target":"head","outcome":"landed","damage":20}'

# Response:
# {"commentary": "Alvarez lands a sharp hook — Garcia needs to tighten that guard.", "latency_ms": 412}
```

### LLM Providers

| Provider | Cost | Latency | Setup |
|---|---|---|---|
| **Groq** (default) | Free tier | ~200ms | `GROQ_API_KEY` from console.groq.com |
| OpenAI | Paid | ~400ms | `OPENAI_API_KEY` |
| Claude | Paid | ~500ms | `ANTHROPIC_API_KEY` |
| Ollama | Free (local) | Varies | `ollama pull qwen2.5:7b` |

## Tests

```bash
python -m pytest tests/unit/ -v   # 27 tests — event lifecycle, queue scoring/decay,
                                   # cooldown timing + override, dominance pendulum,
                                   # pace hysteresis
```

## Tech Stack

Python 3.10+ | Go 1.22+ | PostgreSQL 16 + pgvector | LangChain | Groq / OpenAI / Claude / Ollama | FastAPI | ElevenLabs | pygame | Docker
