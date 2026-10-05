# BoxCommentator Upgrade: Build Plan

Goal: turn BoxCommentator from a working commentary pipeline into a **measured LLM-serving project**: self-hosted Qwen2.5 on vLLM, benchmarked against Nebius Token Factory and Groq, with optimization experiments, a LoRA fine-tune, and live voice. Every resume number must come from a run recorded in `bench/results/`.

Status of this document: written 2026-10-04 after a read of the repo. Items marked **VERIFY** are external facts that change over time and must be checked before relying on them.

---

## 1. Current state (verified in code)

| Area | State | Where |
|---|---|---|
| Six trackers, priority queue, cooldowns, consumer | Built, 27 unit tests | `src/trackers/`, `src/core/` |
| Track A (streaming, interruptible) / Track B (direct) | Built | `src/generation/track_a`, `track_b` |
| LLM clients | Groq (default), Claude, OpenAI, Ollama behind `BaseLLMClient` + `LLMFactory` | `src/generation/llm_interface/` |
| FastAPI service | Built; one global orchestrator | `src/api/` |
| Go gateway | Built (goroutine fan-out, 550 ms timeout, p95 metrics) | `gateway/` |
| RAG | Built on LangChain + pgvector; 4 fighters, 5 history snippets | `src/retrieval/` |
| Docker | Dockerfile + `docker-compose.yml` (postgres, python, go) | repo root |
| TTS | ElevenLabs + pygame | `src/generation/speech_synthesis/` |

### Gaps this plan must close

1. **No knockdown concept.** `Punch` (`src/core/action_buffer/buffer.py`) has no knockdown field; no event is named `knockdown`. Quality checks need ground truth.
2. **No TTFT.** `orchestrator._generate_track_a` joins the whole stream; routes only return total latency.
3. **Outputs are tiny.** Track A max 40 tokens, prompt asks for 10-15 words. Tokens/sec is noise at this length.
4. **Prefix caching would not help.** Prompts start with one short static line; fighter names, event focus and RAG text come early and vary.
5. **RAG is not grounded in practice.** `FakeEmbeddings` is used unless `OPENAI_API_KEY` is set; `set_fighter_names` re-adds documents on every call (duplicates).
6. **Single-fight service.** `src/api/routes.py` keeps one global orchestrator; the route is `async` but calls blocking LLM code.
7. **Unmeasured README claims.** "Sub 600 ms p95" and "20% latency reduction" (README itself says "17% rounds to 20%"). Claude model name in README does not match `llm_config.py`.
8. **No benchmark, serving, fine-tune or Kokoro code exists.**

---

## 2. Decisions

| Decision | Choice | Reason |
|---|---|---|
| Rename `src/` to `app/` | **No**, keep `src/` | Churns every import, test and the Dockerfile; no resume value |
| Keep Go gateway | Keep in repo, **leave out of resume bullets** | It exists and works; new story is inference |
| Backend abstraction | One `OpenAICompatClient(base_url, model, api_key)` | vLLM, Nebius, Groq are all OpenAI-compatible |
| Backend list | `config/backends.yaml` | Single place for url, model, prices |
| Quantization pair | **3B fp16 vs 3B-AWQ** by default | 7B fp16 weights (~15 GB) do not fit one 15 GB T4 with KV cache |
| Self-hosted "cost" | GPU-hour-equivalent price x seconds used | Kaggle is $0, which makes a cost column meaningless |
| Bench execution | Run `bench.py` **inside the Kaggle notebook** | A tunnel (cloudflared/ngrok) adds latency and pollutes TTFT; Kaggle has no inbound networking |
| RAG in benchmarks | Inject stats directly, bypass pgvector | Removes network noise, works offline, deterministic |

---

## 3. Target layout (additions only)

```
config/backends.yaml            # base_url, model, api_key_env, price_in/out per 1M tokens
src/generation/llm_interface/openai_compat_client.py
src/generation/router.py        # per-track backend routing
bench/
  fights.py                     # seeded fight simulator with ground-truth knockdowns
  bench.py                      # runs fights x backends, logs timing
  quality.py                    # knockdown recall + invented-stat checks
  charts.py                     # results -> PNG
  results/                      # CSVs + charts (committed)
serving/
  launch_3b.sh  launch_3b_awq.sh  launch_7b_awq.sh  launch_lora.sh
  kaggle_vllm.ipynb
finetune/
  gen_data.py                   # synthetic examples from big Token Factory model
  data/train.jsonl  data/eval.jsonl
  train_lora.ipynb
voice/
  kokoro_tts.py                 # streaming synth + time-to-first-audio
k8s/                            # optional (kind)
docs/BUILD_PLAN.md
```

---

## 4. Phase 0: make it measurable (about 1 day)

Everything later depends on this. Do it first.

### 0.1 Knockdown in the data model
- Add `knockdown: bool = False` to `Punch`.
- Emit an `Event(type="knockdown", priority>=9.5)` from a tracker (extend `ExcitementTracker` or add a small `KnockdownTracker`) when a punch has `knockdown=True`.
- Confirm Track B fires on it (threshold is 9.0 in `orchestrator._generate_commentary_if_needed`) and that cooldown override (9.5+) lets it through.
- Add unit tests: knockdown punch -> event -> Track B route.

### 0.2 Fight simulator (`bench/fights.py`)
- Seeded (`random.Random(seed)`), reproducible: `make_fight(seed, n_punches, n_knockdowns) -> Fight`.
- A `Fight` holds: fighter names (from `SAMPLE_FIGHTERS`), ordered punches with timestamps, and **ground-truth knockdown indices**.
- Generate 30 fights: a mix of short (60 punches) and long (300+ punches, needed for the context-capping test); 3-5 knockdowns in total per 5 fights at minimum so "no missed knockdowns" is statistically meaningful.
- Use a **virtual clock** instead of `time.time()`/`sleep` so a fight replays in seconds. Check how `ActionBuffer.get_recent_seconds`, `PaceTracker` and `ExcitementTracker` read time and inject a clock where needed.

### 0.3 Stream timing
- Add a timing wrapper around `generate_streaming`: records `t_request`, `t_first_token`, `t_done`, token count (chunk count approximates tokens; prefer `usage` from the API when `stream_options={"include_usage": true}` is available).
- Expose through the orchestrator as a `GenerationRecord(track, backend, ttft_ms, total_ms, n_tokens, text, event_type)` list.

### 0.4 Prompt restructure for prefix caching
- Order each prompt: **static system text, fighter profiles (fixed per fight), then volatile content** (round, event focus, tracker state, recent commentary).
- Put RAG/stat text before volatile fields; keep byte-identical across calls within a fight.
- Add a test asserting two prompts for the same fight share a prefix of at least N characters.
- Add a `benchmark_mode` flag: longer output cap (for example 80-120 tokens) so tokens/sec is meaningful. Keep production caps unchanged.

### 0.5 Per-fight orchestrators
- Replace the global singleton in `routes.py` with a `dict[fight_id, CommentaryOrchestrator]`; add `fight_id` to `PunchRequest`.
- Run blocking generation via `run_in_threadpool` or move to an async client.
- Fix duplicate seeding: seed once per fight (guard with a flag or use deterministic document IDs).

### 0.6 README honesty pass
- Remove "sub 600 ms p95" and "20%" until measured; replace with a placeholder linking to the future results table.
- Fix the Claude model name mismatch.
- State clearly what exists (Go gateway, pgvector) and what is being added.

**Exit criteria:** `python -m bench.fights --seed 1` prints a fight with known knockdowns; tests pass; one simulated fight replays in under 10 s using Groq with timing records captured.

---

## 5. Weekend 1: backends and self-hosting

### 1.1 `OpenAICompatClient`
- Implements `BaseLLMClient` (`generate`, `generate_streaming`, `health_check`).
- Uses the `openai` package with `base_url`/`api_key`; adds `stream_options={"include_usage": True}`.
- Register in `LLMFactory` as `provider="openai_compat"` taking a backend name from `backends.yaml`.

### 1.2 `config/backends.yaml`
```yaml
backends:
  vllm-3b:      {base_url: "http://localhost:8000/v1", model: "Qwen/Qwen2.5-3B-Instruct", api_key_env: "VLLM_KEY", price_per_gpu_hour: 0.35}
  vllm-3b-awq:  {base_url: "...", model: "Qwen/Qwen2.5-3B-Instruct-AWQ", ...}
  vllm-7b-awq:  {base_url: "...", model: "Qwen/Qwen2.5-7B-Instruct-AWQ", ...}
  nebius:       {base_url: "<VERIFY>", model: "<VERIFY>", api_key_env: "NEBIUS_API_KEY", price_in: 0.0, price_out: 0.0}
  groq:         {base_url: "https://api.groq.com/openai/v1", model: "llama-3.3-70b-versatile", api_key_env: "GROQ_API_KEY", price_in: 0.0, price_out: 0.0}
```
Prices are placeholders; fill from each provider's pricing page (**VERIFY**).

### 1.3 Kaggle + vLLM (`serving/`)
- Notebook: enable GPU (T4), `pip install vllm`, launch `vllm serve <model> --dtype half --max-model-len 4096 --gpu-memory-utilization 0.9`.
- **VERIFY** before building: current vLLM T4 support (no bf16, attention backend), AWQ kernel support on compute capability 7.5, and the `--enable-prefix-caching` flag name/default in the installed version.
- Launch scripts per variant so each run is one command and settings are recorded.
- Capture `vllm --version`, GPU, flags into `bench/results/run_meta.json` for every run.

### 1.4 First end-to-end fight
- Run one simulated fight through the Kaggle server inside the notebook; confirm streamed commentary and recorded TTFT.

**Exit criteria:** one fight commentated end to end on your own vLLM server, timing records saved.

---

## 6. Weekend 2: benchmark and optimization (the core)

### 2.1 `bench/bench.py`
- Inputs: backend name(s), fight set, concurrency, flags.
- For each fight, replay punches through a fresh orchestrator, collect `GenerationRecord`s.
- Output one CSV row per generation and a per-run summary:
  - TTFT p50 / p95 (ms), total latency p50 / p95
  - tokens/sec (decode: `n_tokens / (t_done - t_first_token)`)
  - cost per fight (API: tokens x price; self-hosted: GPU-hour price x wall seconds)
  - error/timeout count
- Include a **warm-up** request per run and discard it; run each config 3 times and report the median to dampen noise. Rate-limit-aware retries for Groq (free tier limits, **VERIFY**).

### 2.2 `bench/quality.py`
- **Knockdown recall:** for each ground-truth knockdown, was there a Track B generation attributed to it, and does the text mention a knockdown/down/canvas/on the floor? Report recall and also false-call rate.
- **Invented stats:** extract numeric claims (percentages, records like `60-2-2`, inches, punch counts) from output with regex; flag any not present in the injected stats block or live window stats. Report invented-stat rate per 100 generations.
- Optional small quality check: LLM-judge or human spot-check of 30 samples per backend, to support any "matched 7B quality" claim.

### 2.3 Experiments (run in this order; each builds on the last)

| # | Experiment | Variants | Primary metric | Notes |
|---|---|---|---|---|
| 1 | Baseline | vllm-3b, vllm-7b-awq, nebius, groq | TTFT p50/p95, quality | 30 fights each |
| 2 | Prefix caching | on vs off, same model | TTFT p50 | Requires Phase 0.4 prompt layout |
| 3 | Quantization | 3B fp16 vs 3B-AWQ (7B pair only if it fits / 2xT4) | latency, tokens/sec, knockdown recall | Report quality delta, not just speed |
| 4 | Batching | 1, 4, 16 concurrent fights | TTFT p95, throughput | Needs per-fight orchestrators |
| 5 | Context capping | capped history vs full vs caching, on long fights | TTFT, quality | Long fights only |
| 6 | Routing | knockdown -> fastest, quiet -> smartest vs single model | TTFT on knockdowns, quality overall | See 2.4 |

### 2.4 Routing (`src/generation/router.py`)
- `QueueConsumer` already splits Track A and B. Routing = one backend per track (or per event type).
- Config: `routing: {track_b: vllm-3b, track_a: groq}` in `backends.yaml`.
- Fallback: if the chosen backend errors or exceeds the track timeout, fall back to the next backend, then to the template `EventSynthesizer`.
- Benchmark routed vs best single model on identical fights.

### 2.5 Results
- `bench/charts.py` writes PNGs; commit CSVs and charts in `bench/results/`.

**Exit criteria:** a results table for all six experiments with run metadata; every number reproducible from a committed command.

---

## 7. Weekend 3: fine-tune, voice, write-up

### 3.1 Training data (`finetune/gen_data.py`)
- Run simulator fights, capture **real prompts** the app builds, and have a large Token Factory model write the target commentary (~300 examples; more if budget allows).
- Split by **whole fight** into train/eval (for example 240/60 or 10/3 fights) so eval is unseen.
- Filter: drop outputs with invented stats (reuse `quality.py`), drop over-length ones.
- Format as chat JSONL matching the Qwen chat template.

### 3.2 LoRA training (`finetune/train_lora.ipynb`)
- Qwen2.5-1.5B-Instruct, Transformers + PEFT, r=16, alpha=32, target attention + MLP projections; fp16 on T4 (or 4-bit QLoRA if memory is tight). **VERIFY** library versions in the notebook.
- Save adapter; record training loss and eval loss.

### 3.3 Serve and benchmark the adapter
- `vllm serve Qwen/Qwen2.5-1.5B-Instruct --enable-lora --lora-modules boxer=<adapter>` (**VERIFY** flags).
- Add `vllm-1.5b-lora` to `backends.yaml`; run the same 30 fights; compare against base 1.5B and the 7B/3B models on quality and latency.
- Only claim "matched 7B quality" if `quality.py` metrics and the judged sample support it.

### 3.4 Voice (`voice/kokoro_tts.py`)
- Kokoro TTS (**VERIFY** current package/model names). Synthesize per sentence or chunk and stream audio.
- Log **time-to-first-audio** = commentary text ready -> first audio chunk produced (and optionally -> first sample played).
- Keep the existing `AudioPipeline` interrupt behavior; note it currently buffers whole clips before playback, so streaming playback needs a change if first-audio-to-ear is the metric.
- Keep ElevenLabs as an optional engine behind the same interface.

### 3.5 README
- One results table, one chart, and a plain-English **Recommendation** (which backend for which track, with numbers).
- Link run metadata and reproduction commands.

### 3.6 Optional: kind (`k8s/`)
- Deployment + Service for the Python service, optionally Postgres. Only do it if everything above is done; list Kubernetes on the resume only if completed.

---

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Kaggle session limits (time/quota), no inbound network | Bench inside notebook; save results to `/kaggle/working` and download; script every launch |
| T4 limits (15 GB, no bf16) | 3B pair by default; `--dtype half`; short `--max-model-len` |
| Groq/Nebius rate limits | Backoff, spread runs, record 429 counts, lower concurrency for API backends |
| Noisy timings | Warm-up, 3 repeats, medians, record time of day and run metadata |
| Prefix caching shows no gain | Check prompt prefix layout (0.4) and vLLM cache hit metrics before concluding |
| Small samples for quality claims | Enough knockdowns per run; report counts alongside rates |
| Fake embeddings hide RAG quality | Benchmarks inject stats directly; state RAG limits in README |
| Overclaiming on resume | Fill each `[X]` only from `bench/results/` |

---

## 9. Resume mapping

Heading (suggested): `BoxCommentator, AI Sports Commentator | Python, vLLM, Transformers, FastAPI, Docker`

| Bullet | Backing evidence |
|---|---|
| Self-hosted Qwen2.5 on vLLM, benchmarked vs Nebius and Groq on 30 fights | Experiment 1 table |
| Prefix caching cut p50 TTFT [X to Y] ms | Experiment 2 |
| AWQ cut [Z]% latency, no missed knockdowns | Experiment 3 + `quality.py` recall |
| Routing sent knockdowns to the fastest model | Experiment 6 |
| LoRA 1.5B matched 7B quality at [N]% lower latency | Section 3.3 + quality evidence |
| Live voice, [M] ms time-to-first-audio | Section 3.4 log |

Skills to list: vLLM, Hugging Face Transformers, PEFT/LoRA, AWQ quantization; Kubernetes only if 3.6 is done.

---

## 10. Checklist

`[x]` done and verified, `[~]` built and unit-tested but never run against a real backend/GPU, `[ ]` not done.

- [x] 0.1 Knockdown in model + tests
- [x] 0.2 Seeded fight simulator, 30 fights, virtual clock
- [x] 0.3 Stream timing records
- [x] 0.4 Prefix-friendly prompts + test
- [x] 0.5 Per-fight orchestrators, seeding fix
- [x] 0.6 README honesty pass
- [x] 1.1 `OpenAICompatClient` + factory
- [x] 1.2 `backends.yaml`
- [~] 1.3 Kaggle vLLM launch scripts + notebook (written, not run)
- [ ] 1.4 One fight end to end on own server (needs Kaggle run)
- [~] 2.1 `bench.py` (tested with stub; no real backend run yet)
- [x] 2.2 `quality.py` (knockdown recall + invented stats)
- [ ] 2.3 Experiments 1-6 run and committed (plan + charts built; runs pending)
- [~] 2.4 Router + routed-vs-single benchmark (router built and tested; benchmark pending)
- [~] 3.1 Training data (~300) with held-out fights (generator built, dry-run only)
- [~] 3.2 LoRA trained (training script + notebook written; not run)
- [ ] 3.3 Adapter served and benchmarked (`--plan lora` ready)
- [~] 3.4 Kokoro + time-to-first-audio (built, tested with fake engine; Kokoro not installed/run)
- [ ] 3.5 README results + recommendation (needs real results)
- [~] 3.6 (optional) kind (manifests written, not run)

---

## 11. Progress log

### 2026-10-04: Phase 0 + Weekend 1 backend layer

Built and tested (48 unit tests pass in the project venv):
- `src/core/clock.py` injectable clock; fight logic no longer calls `time.time()` directly, so fights replay on a virtual clock.
- `Punch.knockdown`, `KnockdownTracker` (priority 10.0, 2 s cooldown, bypass at 9.5), wired into the orchestrator.
- `bench/fights.py` seeded simulator (`make_fight`, `make_suite`, `replay`); the 30-fight suite has 60 knockdowns (mix of 60/150/400-punch fights).
- `orchestrator.generation_log` of `GenerationRecord`s: backend, event type and context, TTFT, total time, chunk and token counts. Track B now streams so it has a TTFT.
- Prompt layout: static header and fighter profiles first, `--- LIVE SITUATION ---` and volatile fields last (test asserts a >600 char shared prefix).
- API: one orchestrator per `fight_id`, blocking work in a thread pool with a per-fight lock; Go gateway forwards `fight_id` and `knockdown`.
- `config/backends.yaml`, `src/config/backends.py`, `OpenAICompatClient` (usage, cached tokens, error counts, thread-local usage); factory resolves backend names. Tested against a local fake OpenAI server.
- `bench/quality.py` knockdown recall; `bench/stub_llm.py`.

Findings that change expectations:
- **Call volume is high.** A 30-fight suite with the stub produced 1,575 generations (~52 per fight; 1,467 Track A, 108 Track B). One full sweep of one hosted backend is ~1,575 requests; the Groq free tier will need pacing (`min_interval_s`) and probably several days of quota across experiments. Consider a 10-fight subset for sweeps and the full 30 only for headline runs.
- **The consumer demotion fix does not affect recall.** The orchestrator sends any event >= 9.0 to the Track B generator regardless of the consumer's track label, so recall was 60/60 with and without the fix. It only corrects the `tone` label. Real recall differences will come from model wording, not routing.
- **Recall as implemented is "called + described".** `described` uses a keyword regex over the generated text; with real models, check false negatives by hand on a sample before trusting it.
- `Ollama` and `Claude` clients were not read or changed; they still use the old swallow-errors pattern.

Open items before benchmarking: `bench/bench.py` (per-backend runner, CSV output, repeats, concurrency), invented-stat check, Kaggle notebook and launch scripts, Nebius endpoint and pricing (**VERIFY**), per-token usage for cost.

### 2026-10-04 (later): everything built, nothing run

Added since the first log entry (70+ unit tests, all passing):
- `bench/bench.py` runner, `bench/charts.py` (renders from recorded runs only; partial runs excluded), invented-stat check
- `src/generation/router.py` (per-track routing + fallback), thread-local virtual clock for concurrent fights
- `serving/run_matrix.py` (vLLM variants, plans, server lifecycle), `serving/kaggle_vllm.ipynb`
- `finetune/gen_data.py`, `finetune/train_lora.py`, `finetune/train_lora.ipynb`
- `voice/kokoro_tts.py`, `voice/speak_fight.py`
- Dockerfile curl fix, `.dockerignore`, compose env, `k8s/`, `docs/RUNNING.md`

Bugs found and fixed while building: live stats were dropped when RAG was off; `.env` was never loaded by the
benchmark path; compose healthcheck needed curl; no `.dockerignore` (would bake `venv/` into the image);
real API keys had been pasted into the tracked `.env.example` (moved to the gitignored `.env`, never committed).

Findings: Groq's free tier (about 100K tokens/day on the 70B model) allows only about 3 fights per day, so Groq is
benchmarked on a small subset with the other backends on more. Nebius: about $1 trial credit; card needed only to
top up.

Still needed from the user: choose `NEBIUS_MODEL`, fill Nebius and Groq prices in `config/backends.yaml`, push the
repo so Kaggle can clone it, then run the notebooks.
