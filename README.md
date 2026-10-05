# Box.IO AI Commentator

A real-time boxing commentator. You feed it punches as they happen, it keeps track of how the fight is going, and it talks about it, with a fast reaction when something big lands and a calmer read when a pattern is building.

I started this as a pipeline project and later turned it into a study of how to serve the language model behind it. The second half is where most of the measured results are, so they come first.

## What I measured

Same simulated fights for every backend: 10 seeded fights, about 470 generations per run, one fight at a time unless stated. The simulator knows where every knockdown is, so I can check whether the commentary actually called it. Raw runs are in `bench/results/`, and the table below is `bench/results/results_table.md`.

| Backend | TTFT p50 | TTFT p95 | Tokens/s | Knockdowns described | Invented stats per 100 | Errors |
|---|---|---|---|---|---|---|
| Qwen2.5 1.5B (vLLM) | 96 ms | 106 ms | 64 | 84% | 0.4 | 0% |
| Qwen2.5 3B AWQ (vLLM) | 114 ms | 125 ms | 84 | 32% | 0.0 | 0% |
| Qwen2.5 3B fp16 (vLLM) | 141 ms | 156 ms | 37 | 32% | 0.0 | 0% |
| Qwen2.5 1.5B + my LoRA (vLLM) | 160 ms | 169 ms | 28 | 100% | 0.8 | 0% |
| Qwen2.5 7B AWQ (vLLM) | 216 ms | 246 ms | 43 | 100% | 0.2 | 0% |
| Nebius, Qwen3-30B-A3B | 515 ms | 667 ms | 144 | 95% | 2.5 | 0% |
| Nebius, Qwen3.8 27B | 769 ms | 900 ms | 404 | 95% | 4.9 | 0% |
| Groq, Qwen3.8 27B | 2,436 ms | 3,359 ms | 429 | 83% | 5.0 | 33% |

The self-hosted models ran on a free Kaggle T4. TTFT is time to first token, which is what decides how soon a line of commentary can start. For a live commentator that matters more than raw throughput.

A few things I'd stand behind:

- **Quantization helped on every speed measure.** AWQ on the 3B model cut TTFT p50 from 141 to 114 ms, total p95 from 780 to 357 ms, and more than doubled decode speed (37 to 84 tokens/s). Knockdown coverage didn't change, though it was only 32% either way.
- **Prefix caching helped a lot.** With caching off, TTFT p50 went from 141 ms to 253 ms on the normal prompt. With the whole fight transcript in the prompt, it went from 198 ms (cache on) to 477 ms (off), and p95 from 313 ms to 1,590 ms. These are single runs and the cache-on server may have been warm from earlier work, so read the exact percentages loosely. The direction is not in doubt.
- **Batching trades latency for throughput.** One fight at a time gave 29 tokens/s. Four at once gave 66, sixteen gave 89, and p95 TTFT went from 156 ms to 331 ms.
- **Model size matters for quality more than I expected.** The 3B models only described 32% of knockdowns. The 7B described all of them. The plain 1.5B landed at 84%, which surprised me (more on that below).
- **The fine-tune worked on coverage, not on speed.** I trained a LoRA adapter on 300 examples written by a much bigger model. The 1.5B with the adapter described 100% of knockdowns, same as the 7B, with a lower TTFT (160 ms vs 216 ms). But it decoded at 28 tokens/s against 64 for the plain 1.5B. I served the adapter unmerged, which costs speed, so that number is probably worse than it needs to be. I haven't tested a merged adapter. It also invented slightly more stats than the base model (0.8 vs 0.4 per 100), so it isn't strictly better.

Things I'd be careful about:

- Each number comes from 10 fights and one run. Knockdown rates rest on about 19 events per run. Treat differences of a few points as noise.
- The 1.5B beating the 3B on knockdown coverage is odd. I haven't dug into why, and it may partly be how the checker matches wording.
- Groq's free tier returned errors on a third of requests. I think it's rate limiting but I never logged the error text, so that's a guess. Its latency numbers come from the requests that did succeed.
- Hosted prices in `config/backends.yaml` are estimates I haven't verified, so the cost-per-fight column isn't something to quote.

### Recommendation

If you need to run this yourself on one small GPU, use the 7B AWQ model with prefix caching on. It gets every knockdown, stays under 250 ms to first token, and fits on a T4. If first-token speed matters more than coverage, the LoRA-tuned 1.5B is the one to look at, ideally merged into the base weights first. I wouldn't use the plain 3B models for this task at all. And if you don't want to run a GPU, Nebius worked fine, but its first token took roughly 2.5 to 3.5 times longer than the local 7B AWQ (515 to 769 ms against 216 ms).

Charts: `bench/results/ttft_by_backend.png`, `quantization.png`, `caching.png`, `batching.png`.

## How the commentator works

Each punch goes into a short rolling buffer, and six small trackers watch it:

| Tracker | What it watches |
|---|---|
| Dominance | Who is controlling the fight. A five-state machine (`P2_DOM`, `P2_EDGE`, `EVEN`, `P1_EDGE`, `P1_DOM`), so it can't jump from even to dominant without passing through edge. |
| Pace | Punch rate, with separate thresholds for entering and leaving a state so it doesn't flicker around a boundary. |
| Momentum | The last 10 punches against the 10 before them. |
| Excitement | How dense the action is, plus a 4 second lull detector for filler commentary. |
| Targets | Head versus body landing ratio, to spot a change in plan. |
| Round context | Early, middle or late in the round. |

Plus a knockdown tracker that fires a top-priority event once per knockdown and ignores cooldowns.

Trackers emit events, and events go into a priority queue where the score decays with age (`priority * 2^(-age/5s)`), so a stale event loses to a fresh one. A cooldown manager stops the commentator repeating itself, except for knockdowns, which always get through.

Two generators pick events off the queue:

- **Track A** is the analyst. It streams a line about a pattern, and can be cut off.
- **Track B** is the reaction. It's for knockdowns and big combinations, and it can't be interrupted. If a high priority event (9.0 or above) arrives while Track A is mid-sentence, Track A is told to stop and Track B takes over.

Prompts are laid out with the stable parts first (rules, fighter profiles) and the live situation last. That's what lets the server reuse its prefix cache.

If the model is down, a template based fallback keeps commentary going instead of crashing.

### Grounding

The model used to make up stats ("Garcia has a 70% KO rate", with nothing behind it). Fighter profiles and past fight summaries now live in pgvector, and each request pulls the few most relevant ones into the prompt. The benchmark checks this too: any number in the output that wasn't in the prompt counts as an invented stat.

### Backends

Every model is an entry in `config/backends.yaml` with a base URL and a model name. vLLM, Nebius and Groq all speak the OpenAI API, so switching is a config change. Set `COMMENTARY_BACKEND` to pick one. Claude, OpenAI and Ollama clients also exist from the original version. A router can send Track B to the fastest backend and Track A to the best one, with fallback if one fails. I built it and tested it with stubs, but I never ran the routed benchmark.

### The Go gateway

A small Go service sits in front of the Python one. It sends each punch to the Python service, logs telemetry in a separate goroutine, and gives up on commentary after 550 ms, returning a 504. It also keeps a rolling p95 of the last 1000 requests.

I haven't benchmarked the gateway, so I'm not claiming a latency win from it. The telemetry goroutine is a log write, so it saves very little. The timeout is the part that matters, because it caps the worst case.

## Layout

```
mock_fight.py        Python only demo, simulates a round
bench/               simulator, benchmark runner, quality checks, charts, results
serving/             vLLM variants and the Kaggle notebook that runs them
finetune/            training data generation and LoRA training
voice/               Kokoro text to speech (written, not part of the results)
k8s/                 kind manifests (written, never deployed)
config/              backends.yaml
gateway/             Go gateway (main.go, handlers, parallel, telemetry, client)
docs/                build plan, progress notes, how to run things
src/
  api/               FastAPI service, one orchestrator per fight id
  core/              buffer, events, priority queue, cooldowns, consumer, context builder, orchestrator
  generation/        LLM clients, Track A and B, timing, router, speech
  retrieval/         pgvector store, fighter stats, historical search, live stats
  synthesis/         template fallback, contradiction detector
  trackers/          the trackers above
tests/unit/
```

## Running it

Python only:

```bash
pip install -r requirements.txt
cp .env.example .env     # add GROQ_API_KEY (console.groq.com)
python mock_fight.py
```

Full stack with the gateway and pgvector:

```bash
cp .env.example .env
docker compose up
curl -X POST http://localhost:8080/api/v1/punch \
  -H "Content-Type: application/json" \
  -d '{"attacker":1,"punch_type":"hook","target":"head","outcome":"landed","damage":20}'
```

To repeat the benchmarks, see [`docs/RUNNING.md`](docs/RUNNING.md). The tool is `python -m bench.bench`, and `python -m serving.run_matrix` starts the vLLM servers on Kaggle. Both have a dry run mode that needs no network. Put real keys in `.env` only, which is gitignored.

Tests: `python -m pytest tests/unit -q`. That's 99 tests (98 pass, 1 skipped because Kokoro isn't installed). They cover the event lifecycle, queue decay, cooldowns, the trackers, the knockdown path, the benchmark tooling and the API.

## What's not done

- Kokoro voice and time to first audio exist as code but I dropped them from the project, so there are no results for them.
- The kind deployment was never run.
- The routing and context capping experiments were never run.
- I haven't tested a merged LoRA adapter, which is the obvious next step for speed.

## Stack

Python, Go, FastAPI, PostgreSQL with pgvector, LangChain, vLLM, Transformers and PEFT, Docker. Self-hosted runs were on a Kaggle T4.
