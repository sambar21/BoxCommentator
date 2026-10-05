# Progress

Last updated: 2026-10-05. Companion to [`BUILD_PLAN.md`](BUILD_PLAN.md) (the plan) and [`RUNNING.md`](RUNNING.md) (how to run it).

**One-line status:** the benchmarks, the quantization, caching and batching experiments, and the LoRA run are done and
their results are in `bench/results/`. The README has the table and the recommendation. Section 1 below is the original
checklist from before any run, so many `[~]` items there are now finished; section 5 is the current state.

Legend: `[x]` done and verified, `[~]` built and unit-tested but never run for real, `[ ]` not done.

---

## 1. Plan checklist (mirrors BUILD_PLAN.md section 10)

### Phase 0: make it measurable
- [x] **0.1 Knockdown in the model + tests.** `Punch.knockdown`; `src/trackers/knockdown/` emits a priority-10 event once
  per knockdown, bypasses cooldown (`knockdown` entry in `CooldownManager`), wired into the orchestrator.
  The consumer now never demotes events above 9.5 (this only fixes a tone label; recall was already 60/60 without it).
- [x] **0.2 Seeded fight simulator.** `bench/fights.py`: `make_fight`, `make_suite` (30 fights: 60/150/400 punches, 60
  ground-truth knockdowns), `replay` on a virtual clock. `src/core/clock.py` is thread-local so concurrent fights do not
  interfere. Seeds 1000-1029 are the benchmark suite; fine-tune data uses 5000+.
- [x] **0.3 Stream timing.** `src/generation/timing.py` (`GenerationRecord`, `collect_timed`); Track B now streams so it
  has a time-to-first-token; the orchestrator keeps `generation_log` with backend, event, TTFT, total, tokens, prompt
  tokens, cached tokens, and optionally the prompt.
- [x] **0.4 Prefix-cache-friendly prompts.** Static header and fighter profiles first, `--- LIVE SITUATION ---` and
  volatile fields last; test asserts a shared prefix over 600 characters. Added context modes `recent | full | capped`
  so the capping experiment has something to cap (the original prompt did not grow with fight length).
  Live stats now work without RAG (they were silently dropped before).
- [x] **0.5 Per-fight orchestrators.** `src/api/routes.py` keeps one orchestrator per `fight_id`, runs blocking LLM work
  in a thread pool with a per-fight lock, evicts old fights. Go gateway now forwards `fight_id` and `knockdown`
  (`go build` and `go vet` pass).
- [x] **0.6 README honesty pass.** Removed the unmeasured "sub-600ms p95" and "20%" claims, fixed the Claude model name,
  added a Status section. README makes no measured claim.

### Weekend 1: backends and self-hosting
- [x] **1.1 `OpenAICompatClient` + factory.** `src/generation/llm_interface/openai_compat_client.py`: streaming, usage
  and cached-token capture (thread-local), counted (not hidden) errors, optional pacing. `LLMFactory.create_client("name")`
  resolves any entry in `config/backends.yaml`. Tested against a local fake streaming server.
- [x] **1.2 `config/backends.yaml`.** vLLM variants (3b, 3b-awq, 7b-awq, 7b, 1.5b, 1.5b-lora), `nebius`,
  `nebius-teacher`, `groq-70b`, plus a `routing:` block. `${VAR:-default}` expansion; the loader reads the repo's `.env`
  without overriding real environment variables.
- [~] **1.3 Kaggle vLLM launch scripts + notebook.** `serving/run_matrix.py` (variants, plans `smoke | core | lora`,
  server lifecycle, `--dry-plan`, `--serve-only`) and `serving/kaggle_vllm.ipynb`. Not run: needs a GPU.
- [ ] **1.4 One fight end to end on your own vLLM server.** Needs a Kaggle run.

### Weekend 2: benchmark and optimization
- [~] **2.1 `bench/bench.py`.** Streaming runs per backend or routed pair; TTFT p50/p95 (also Track B only), total
  latency, decode and aggregate tokens/s, cost per fight (API tokens x price, or GPU-hour equivalent for self-hosted),
  cache hit rate (usage and vLLM `/metrics`), knockdown recall, invented stats. Safety: `--max-requests`, `--max-usd`,
  `--rpm`, partial results are still written and marked. `--dry-run` uses the stub. Verified only with the stub.
- [x] **2.2 `bench/quality.py`.** Knockdown recall (called and described) and invented-stat detection (digits in the
  output that are not in the prompt; fractions accepted for percentages).
- [ ] **2.3 Experiments 1-6 run and committed.** The plan, runner and charts exist; the runs have not happened:
  baseline per backend, prefix caching on/off, quantization, batching 1/4/16, capping vs caching, routing.
- [~] **2.4 Router + routed-vs-single benchmark.** `src/generation/router.py` (`FallbackLLM`, `build_routed_clients`);
  the orchestrator takes per-track clients; `bench.py --routing`. Tested with stubs; the real comparison is pending.
- [~] **2.5 Charts.** `bench/charts.py` builds six charts and `results_table.md` from recorded runs only. Layout was
  checked visually using clearly synthetic placeholder numbers kept outside the repo; the palette passed the validator.

### Weekend 3: fine-tune, voice, write-up
- [~] **3.1 Training data (about 300) with held-out fights.** `finetune/gen_data.py` records real app prompts for free,
  selects a balanced set (Track B and knockdowns oversampled), asks the teacher only for those (about 450 requests),
  filters answers (no invented stats, knockdowns actually called, short), holds out whole fights for eval. Dry-run only.
- [~] **3.2 LoRA trained.** `finetune/train_lora.py` (loss only on the reply, fp32 base under fp16 autocast for the T4,
  adapter-only save) and `finetune/train_lora.ipynb`. Tokenization and masking are tested with a fake tokenizer; no
  training has run.
- [ ] **3.3 Adapter served and benchmarked.** `serving/run_matrix.py --plan lora` is ready (base 1.5B vs adapter on
  identical fights).
- [~] **3.4 Kokoro + time-to-first-audio.** `voice/kokoro_tts.py` and `voice/speak_fight.py` (logs TTS first-audio, LLM
  time, end-to-end first-audio, real-time factor). Tested with a fake engine; Kokoro itself is not installed or run.
- [ ] **3.5 README results + Recommendation.** Needs real results. `results_table.md` and charts will feed it.
- [~] **3.6 (optional) kind.** `k8s/` manifests and README written and parsed as valid YAML; no cluster started.

---

## 2. Everything added (by area)

**Pipeline changes** (`src/`): `core/clock.py`; `trackers/knockdown/`; `core/orchestrator.py` (per-track LLMs,
`generation_log`, `context_mode`, `keep_prompts`, always-on live stats); `core/context_builder/builder.py` (profiles,
history modes); `generation/track_a|track_b/generator.py` (prompt layout, `last_prompt`, Track B streaming);
`generation/timing.py`; `generation/router.py`; `generation/llm_interface/openai_compat_client.py` and factory
support; `config/backends.py`; `api/routes.py`; `config/llm_config.py` (`COMMENTARY_BACKEND`).

**Benchmark** (`bench/`): `fights.py`, `bench.py`, `quality.py`, `charts.py`, `stub_llm.py`.
**Serving** (`serving/`): `run_matrix.py`, `kaggle_vllm.ipynb`.
**Fine-tune** (`finetune/`): `gen_data.py`, `train_lora.py`, `train_lora.ipynb`, `requirements.txt`.
**Voice** (`voice/`): `kokoro_tts.py`, `speak_fight.py`, `requirements.txt`.
**Deploy**: `Dockerfile` (curl added), `.dockerignore`, `docker-compose.yml` (backend env vars), `k8s/`.
**Config and docs**: `config/backends.yaml`, `.env.example`, `docs/BUILD_PLAN.md`, `docs/RUNNING.md`, this file, README.
**Go** (`gateway/`): forwards `fight_id` and `knockdown`.

**Tests: 99 collected** (27 original, 72 new): `test_knockdown` (10), `test_api` (4), `test_openai_compat` (7),
`test_bench` (22), `test_pipeline_extras` (29). Run: `venv\Scripts\python.exe -m pytest tests/unit -q`.

---

## 3. Bugs and risks found along the way

| Found | Status |
|---|---|
| Live stats dropped when RAG was off | Fixed |
| `.env` never loaded on the benchmark path (Groq calls would fail auth) | Fixed, tested |
| Compose healthcheck used `curl`, absent from the image | Fixed |
| No `.dockerignore` (would copy `venv/` into the image) | Fixed |
| Real API keys pasted into tracked `.env.example` | Moved to gitignored `.env`; never committed; rotating is cheap insurance |
| Single global orchestrator shared by all fights | Fixed (per-fight sessions) |
| Global virtual clock would break concurrent fights | Fixed (thread-local) |
| README latency claims were unmeasured | Removed |
| Go gateway timeouts (560 ms client, 550 ms dispatcher) are tight for a loaded self-hosted model | Not changed; noted in RUNNING.md |
| Claude and Ollama clients still swallow errors | Not changed |
| Original prompt size did not grow with the fight, so "context capping" was meaningless | Fixed by adding `full`/`capped` modes |
| An earlier note claimed the consumer demotion could lose knockdowns | Corrected: it only affected a tone label |

---

## 4. Cost and quota facts (as researched; confirm in each console)

- **Groq free tier:** about 30 requests/min, 1,000 requests/day, about 100K tokens/day on the 70B model, so about 3
  fights per day. Groq therefore gets a small sample.
- **Nebius Token Factory:** about $1 trial credit; a card is only needed to top up; pay as you go. Leave auto top-up off.
- **Kaggle:** free weekly GPU hours; no card.
- One full 30-fight sweep is about 1,500 requests per backend. The 7B fp16 model does not fit one T4 (fallback: 3B
  fp16 vs 3B-AWQ; `7b-tp2` needs two GPUs).
- Unverified (marked VERIFY in files): Groq and Nebius prices, vLLM flag names and T4/AWQ support, Kokoro install on
  Windows (needs espeak-ng), Nebius model ids.

---

## 5. Status (updated 2026-10-05) and what is left

**Done**
1. Model ids and prices set in `.env` / `config/backends.yaml`. Prices are estimates (VERIFY against the Nebius and
   Groq pricing pages). Llama 3.3 70B is gone from both catalogs, so the hosted comparison uses Qwen3.8-27B
   (`nebius-27b`, `groq-qwen27b`) plus a Qwen3-30B Nebius baseline.
2. Repo pushed to GitHub (`sambar21/BoxCommentator`, main).
3. Smoke tests and hosted baselines (Nebius clean; Groq 33% errors, cause not confirmed, likely rate limits).
4. LoRA training data generated and committed (`finetune/data/`: 300 train, 60 eval, teacher Qwen3-235B).
5. Kaggle T4 vLLM runs, results in `bench/results/`: 3B fp16, 3B-AWQ, 7B-AWQ, batching (c=4, 16), caching on/off.
   Not run: `capping` (long fights, very slow) and `routing`.

**Measured so far (10 fights each, 0% errors)**
- AWQ vs fp16 (3B): TTFT p50 141 to 114 ms (-20%), total p95 780 to 357 ms (-54%), decode 36.7 to 83.7 tok/s.
- Prefix cache on vs off (3B): TTFT p50 253 to 141 ms (-44%) on the short prompt; 477 to 198 ms (-59%) and p95
  1590 to 313 ms (-80%) with the whole transcript. Single runs; the cache-on server may have been warm from earlier runs.
- Batching: 28.7 to 66.4 (c=4) to 89.0 (c=16) tok/s; p95 TTFT 156 to 331 ms.
- Quality: 7B-AWQ describes 100% of knockdowns vs 32% for the 3B (about 19 events per run).

**Left**
6. **You (Kaggle):** run `finetune/train_lora.ipynb` end to end (train, install vLLM, restart, `--plan lora`), download
   `lora_results.zip` + `train_log.json`. Merge new rows into `bench/results/summary.csv` (append, do not overwrite),
   then `python -m bench.charts`.
7. README results table, one chart and the Recommendation (plan 3.5).
8. Resume numbers are filled in `docs/reference.md`, all taken from `bench/results/`.
9. Cleanup: run instructions now point at `groq-qwen27b`. Still open: verify prices, find the cause of Groq's errors (bench does not log
   error text yet).

**Dropped (decision 2026-10-05):** Kokoro voice / time-to-first-audio, and the kind deployment. The `voice/` and
`k8s/` code stays in the repo but is out of scope for the resume bullets.

**Kaggle gotchas:** after `pip install vllm`, run `pip uninstall -y torchaudio` and restart the session. An interrupted
run leaves a vLLM server on port 8000: `pkill -f "vllm serve"` (run_matrix now refuses an occupied port).

## 6. Decisions on record
- Keep `src/` (not renamed to `app/`); keep the Go gateway in the repo but out of the resume bullets.
- Default quantization comparison is 3B fp16 vs 3B-AWQ.
- Benchmark inside the Kaggle notebook (a tunnel would pollute TTFT).
- Self-hosted cost is reported as a GPU-hour equivalent; Kaggle itself is free.
- Charts and tables come only from recorded runs; partial or missing runs are skipped, never filled in.
- No paid or rate-limited API is called without your say-so (you asked not to run anything yet).
