# Running the benchmark, fine-tune and voice

Everything below was built and tested with a stub model first. The Kaggle and hosted runs have since been done, and
the results are in `bench/results/`.
Commands are ordered cheapest and safest first. Do not skip the dry runs.

## 0. Setup (local)

```bash
python -m venv venv && venv\Scripts\activate          # Windows
pip install -r requirements.txt
cp .env.example .env                                  # real keys go in .env only (gitignored)
python -m pytest tests/unit -q                        # all green before spending anything
```

`.env` needs `GROQ_API_KEY` and `NEBIUS_API_KEY`; set `NEBIUS_MODEL` to a model id from the Token Factory
catalog. Check what is configured: `python -m bench.bench --list`.

## 1. Cost and quota (read this first)

| Service | What limits you | How the tools protect you |
|---|---|---|
| Groq free tier | about 30 requests/min and about 100K tokens/day on the 70B model (VERIFY current limits) | `--rpm 25 --max-requests 150`: about 3 fights per day |
| Nebius Token Factory | pay as you go; about $1 trial credit; a card is needed only to top up | `--max-usd`; **leave auto top-up off** |
| Kaggle | free weekly GPU hours; no card | run one notebook section at a time |

One 30-fight sweep is about 1,500 LLM requests per backend. `--max-usd` uses the prices in `config/backends.yaml`,
which are placeholders until you fill them in from the provider's pricing page; without real prices the dollar cap
cannot track spend, so also pass `--max-requests`.

## 2. Dry runs (free, no network)

```bash
python -m bench.bench --dry-run --fights 3 --out-dir tmp_dry
python -m finetune.gen_data --dry-run --out-dir tmp_ft
python -m voice.speak_fight --dry-run
python -m serving.run_matrix --plan core --dry-plan
```

## 3. First real call (tiny)

```bash
python -m bench.bench --backend groq-qwen27b --fights 1 --kind short --max-requests 20 --rpm 20 --tag smoke
```

## 4. Self-hosted experiments (Kaggle)

Open `serving/kaggle_vllm.ipynb` on Kaggle (GPU T4, Internet on, secrets set) and run the sections in order:
plan, smoke test, core experiments, hosted baselines, routing, charts + zip. Download `bench_results.zip`, unzip
into `bench/results/`, then `python -m bench.charts` locally to rebuild charts and `results_table.md`.

The core plan benchmarks:

| Experiment | Runs |
|---|---|
| Quantization | `3b` fp16 vs `3b-awq` vs `7b-awq` |
| Prefix caching | `3b` vs `3b-nocache`, short prompts and growing transcripts |
| Batching | 1, 4 and 16 concurrent fights |
| Capping vs caching | long fights: full transcript (cached) vs capped transcript, with and without cache |
| Routing | knockdowns to the fastest backend, analysis to the strongest, vs the best single model |

fp16 7B does not fit one T4; the optional `7b-tp2` variant needs two GPUs.

## 5. Fine-tune

```bash
python -m finetune.gen_data --teacher nebius-teacher --max-usd 0.50      # about 450 teacher requests, locally
git add finetune/data && git commit -m "Distilled commentary data"
```
Then `finetune/train_lora.ipynb` on Kaggle trains the adapter. Serve it with
`python -m serving.run_matrix --plan lora --adapter <path>` in a fresh session; the plan runs the base 1.5B and the
adapter on identical fights.

## 6. Voice

```bash
pip install -r voice/requirements.txt          # plus the espeak-ng system package
python -m voice.speak_fight --backend groq-qwen27b --fights 1 --max-lines 30 --play
```
Reports time-to-first-audio (TTS only) and end-to-end first-audio (LLM + TTS).

## 7. Deploy to kind (optional)

See `k8s/README.md`. Not yet run.

## What the numbers mean

- **TTFT**: request sent to first streamed token, measured per generation. Fight logic runs on a virtual clock,
  so only LLM time is real.
- **Knockdowns called**: the share of ground-truth knockdowns with a Track B line that reads as a knockdown call
  (keyword check; spot-check a sample of `generations.csv` by hand).
- **Invented stats / 100**: generations containing a number that is not in the prompt (digits only; arithmetic the model
  does itself counts as invented).
- **$/fight**: hosted = tokens x price; self-hosted = GPU-hour-equivalent price x wall time (Kaggle itself is free).
- Tokens/s is only reported for replies of 8+ tokens; commentary lines are short, so treat it as indicative.
- Runs cut short by a cap are marked `(partial)` and are excluded from charts.

## Known limits

- The Go gateway's HTTP client times out at 560 ms and the dispatcher at 550 ms, which is tighter than a cold or
  loaded self-hosted model; raise both if you put vLLM behind the gateway.
- The Claude and Ollama clients were not changed and still hide errors (return an empty string).
- Prefix-cache hit rates are scraped from vLLM's `/metrics`; metric names vary by version, so a missing value means
  "not reported", not zero.
