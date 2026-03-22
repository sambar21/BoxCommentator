# Box.IO AI Commentator

**A real-time AI commentary engine that watches a boxing match punch-by-punch and calls the fight like a broadcast analyst.**

Feed it raw fight data. Six statistical trackers build a living model of the fight. A priority queue with exponential time-decay surfaces the most interesting events. A dual-track LLM generator produces commentary -- analytical breakdowns for evolving patterns, instant reactions for explosive moments -- with one track able to interrupt the other mid-sentence. Optional ElevenLabs TTS streams it out as voice.

---

## How It Works

A punch lands. Within one pipeline tick:

1. **Buffer** ingests it into a circular `deque` (last 20 actions, O(1) append)
2. **Six trackers** independently analyze the buffer and emit typed events:
   - *Dominance* -- pendulum state machine over a doubly-linked list (`P2_DOM <-> P2_EDGE <-> EVEN <-> P1_EDGE <-> P1_DOM`)
   - *Pace* -- punch frequency with hysteresis (separate enter/exit thresholds to prevent oscillation)
   - *Momentum* -- sliding-window comparison of recent vs. prior performance
   - *Excitement* -- action density with idle-timeout detection for lull commentary
   - *Targets* -- head/body landing ratio to detect strategic shifts
   - *Round Context* -- early/mid/late narrative framing
3. **Cooldown Manager** deduplicates events per type, with priority-based override for critical moments
4. **Priority Queue** scores each event: `score = priority * 2^(-age / half_life)`, so fresh events always surface first
5. **Queue Consumer** pops the top events and routes them:
   - **Track A** (analytical, streaming, interruptible) for strategy and pattern commentary
   - **Track B** (urgent, direct, fast) for knockdowns and big combos -- can interrupt Track A mid-generation
   - Maintains a 60/40 content balance between tracks
6. **Context Builder** packages tracker states, recent actions, fighter names, round phase, and prior commentary into a structured LLM prompt
7. **LLM** generates the line (Groq, OpenAI, Claude, or Ollama -- swappable via factory pattern behind a common `BaseLLMClient` interface)
8. **TTS + Audio Pipeline** (optional) streams ElevenLabs audio through pygame with interrupt support

If the LLM is unavailable, the system falls back to template-based generation via `EventSynthesizer` -- no crash, no silence.

```
Punch ─> ActionBuffer ─> Trackers (6x) ─> Events
                                              |
                         CooldownManager <────┘
                              |
                              v
                     PriorityQueue (scored, sorted, pruned)
                              |
                              v
                       QueueConsumer (route + balance)
                         /          \
                    Track A       Track B
                  (streaming)   (immediate)
                     \            /
                   ContextBuilder ─> LLM ─> Commentary ─> TTS ─> Audio
```

---

## Why These Design Choices

| Problem | Solution | Why |
|---------|----------|-----|
| Dominance isn't binary -- fighters drift in and out of control | Pendulum state machine (doubly-linked list of 5 nodes) | Enforces ordered transitions; can't jump from "even" to "dominating" without passing through "edge" |
| Pace tracker oscillates near thresholds | Hysteresis -- different entry/exit thresholds per state | A standard technique from signal processing; eliminates flip-flopping at boundary values |
| Old events clog the queue and produce stale commentary | Recency-weighted scoring with exponential half-life decay | `2^(-age/5)` means a 10-second-old event scores at 25% of a fresh one |
| Commentator repeats itself ("Player 1 taking control" every 2 seconds) | Per-type cooldown timers with priority override | Blocks repeats, but a 9.5+ priority event bypasses cooldown -- because a knockdown always gets called |
| Need both thoughtful analysis and instant reactions | Dual-track generation with interrupt protocol | Track A streams for 300ms; if a Track B event fires, it cancels mid-stream and takes over |
| Tracker states can contradict each other | Contradiction detector synthesizes tension events | "P1 dominating but P2 building momentum" becomes its own high-priority narrative event |
| Switching LLM providers shouldn't require code changes | Abstract `BaseLLMClient` + factory pattern | Swap `PROVIDER = "groq"` to `"claude"` in config; zero code changes needed |
| LLM goes down mid-fight | Template-based `EventSynthesizer` fallback | Graceful degradation -- commentary quality drops, but the system never stops |

---

## Project Structure

```
.
├── mock_fight.py                       # Entry point -- simulates a round
├── requirements.txt
├── src/
│   ├── config/
│   │   ├── llm_config.py              # Provider selection + model params
│   │   └── tts_config.py              # ElevenLabs voice settings
│   ├── core/
│   │   ├── orchestrator.py             # Central pipeline coordinator
│   │   ├── action_buffer/              # Circular deque (last 20 punches)
│   │   ├── context_builder/            # Assembles LLM prompt from state
│   │   ├── cooldown/                   # Per-type cooldown with override
│   │   ├── events/                     # Event dataclass
│   │   ├── priority_queue/             # Scored queue with decay + staleness pruning
│   │   └── queue_consumer/             # Track A/B routing + balance
│   ├── generation/
│   │   ├── llm_interface/              # BaseLLMClient, 4 providers, factory
│   │   ├── speech_synthesis/           # TTS engine + pygame audio pipeline
│   │   ├── track_a/                    # Analytical (streaming, interruptible)
│   │   └── track_b/                    # Urgent (direct, fast)
│   ├── synthesis/
│   │   ├── aggregator/                 # Template fallback when LLM unavailable
│   │   └── contradiction_detector/     # Cross-tracker narrative tension
│   └── trackers/
│       ├── dominance/                  # Pendulum state machine
│       ├── excitement/                 # Action density + idle timeout
│       ├── momentum/                   # Sliding-window comparison
│       ├── pace/                       # Tempo with hysteresis
│       ├── round_context/              # Phase-aware narrative framing
│       └── targets/                    # Head/body targeting patterns
└── tests/unit/                         # 27 tests across 5 modules
```

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Add at minimum: GROQ_API_KEY (free at console.groq.com)
python mock_fight.py
```

Runs a 60-second simulated round. Punches stream in, trackers update, and AI commentary appears when the pipeline triggers.

For voice output (requires an ElevenLabs key in `.env`):
```bash
python src/demos/demo_with_voice.py
```

### LLM Providers

| Provider | Cost | Latency | Setup |
|----------|------|---------|-------|
| **Groq** (default) | Free tier | ~200ms | [console.groq.com](https://console.groq.com) |
| OpenAI | Paid | ~400ms | [platform.openai.com](https://platform.openai.com) |
| Claude | Paid | ~500ms | [console.anthropic.com](https://console.anthropic.com) |
| Ollama | Free (local) | Varies | [ollama.com](https://ollama.com) + `ollama pull qwen2.5:7b` |

Switch providers in one line:
```python
# src/config/llm_config.py
PROVIDER = "groq"  # "groq" | "openai" | "claude" | "ollama"
```

---

## Tests

```bash
python -m pytest tests/unit/ -v   # 27 tests, all passing
```

Covers: event lifecycle, priority queue scoring/decay/staleness, cooldown timing and override, dominance pendulum transitions, pace hysteresis.

---

## Tech Stack

Python 3.10+ | Groq / OpenAI / Claude / Ollama | ElevenLabs | pygame
