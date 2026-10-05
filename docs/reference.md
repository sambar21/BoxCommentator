Build order

Weekend 1: foundation + self-hosting

Fix the README so it matches reality, add FastAPI and a Dockerfile, and drop Go from the heading unless the gateway exists. (Gap 10)
Set up one config file that lists backends: your vLLM server, Token Factory, and Groq. Each entry is just a base_url plus a model name, since all three are OpenAI-compatible. (Gap 9)
On Kaggle, launch vLLM serving Qwen2.5-3B and Qwen2.5-7B-AWQ, and get one fight commentated end to end through your own server. (Gap 2)

Weekend 2: benchmark + optimization (the core)

Write bench.py: 30 fights, each backend, streaming responses. Log time-to-first-token (p50/p95), tokens/sec, and cost per fight. (Gap 3)
Write quality.py: did it call every knockdown, and did it say any stat that isn't in your RAG data? (Gap 3)
Add these runs to the benchmark (Gaps 4, 5):
Quantization: 7B full vs. 7B-AWQ. If the full 7B doesn't fit on a T4, compare 3B full vs. 3B-AWQ instead.
Batching: 1, 4 and 16 fights at once.
Caching: prefix caching on vs. off.
Capping: context capping vs. caching on long fights.
Add routing to the app: knockdowns go to the fastest backend, quiet stretches go to the smartest one. Benchmark routed vs. single-model. (Gap 4)

Weekend 3: fine-tune, voice, write-up

Generate about 300 commentary examples with a big Token Factory model, LoRA-tune Qwen2.5-1.5B with Transformers + PEFT on Kaggle, serve the adapter in vLLM, and add it to the benchmark. (Gaps 1, 7)
Pipe the output into Kokoro so the commentator speaks, and log time-to-first-audio. (Gap 6)
Write the README: one results table, one chart, and a "Recommendation" section in plain English. (Gap 8)
Optional: deploy to kind. (Gap 11)
Resume after this

Your project heading becomes:

BoxCommentator, AI Sports Commentator | Python, vLLM, Transformers, FastAPI, Docker

And four bullets:

Self-hosted Qwen2.5 with vLLM and benchmarked it against Nebius Token Factory and Groq on 30 simulated fights: time-to-first-token, throughput, cost and accuracy.
Prefix caching cut p50 time-to-first-token [X→Y] ms; AWQ quantization cut [Z]% latency with no missed knockdowns; routing sent knockdowns to the fastest model.
LoRA-tuned a 1.5B model (Transformers + PEFT) that matched 7B quality at [N]% lower latency.
Added live voice (Kokoro text-to-speech) with [M] ms time-to-first-audio.