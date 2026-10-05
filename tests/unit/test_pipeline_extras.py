"""Tests for charts, serving plan, fine-tune data/tokenization, voice timing and deploy manifests."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import csv
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


# ── serving plan ──────────────────────────────────────────────────────────────

from serving.run_matrix import PLANS, VARIANTS, vllm_command
from src.config.backends import load_backends


def test_every_plan_variant_and_backend_exists():
    backends = load_backends()
    for plan, variants in PLANS.items():
        for name in variants:
            assert name in VARIANTS, f"{plan}: unknown variant {name}"
            assert VARIANTS[name].backend in backends, f"{name}: backend not in backends.yaml"


def test_variant_models_match_backend_models():
    """The model a server is started with must be the model the benchmark asks for."""
    backends = load_backends()
    for name, v in VARIANTS.items():
        expected = backends[v.backend].model
        if "lora" in name:
            assert expected == "boxer"          # adapter is addressed by its served name
        else:
            assert v.model == expected, name


def test_vllm_command_flags():
    cmd = vllm_command("3b-awq", port=8123)
    assert cmd[:3] == ["vllm", "serve", "Qwen/Qwen2.5-3B-Instruct-AWQ"]
    assert "8123" in cmd and "--quantization" in cmd and "awq" in cmd
    assert "--no-enable-prefix-caching" in vllm_command("3b-nocache")
    assert "--enable-prefix-caching" in vllm_command("3b")
    assert "half" in vllm_command("3b")                                   # T4 has no bf16


def test_lora_variant_requires_adapter():
    with pytest.raises(ValueError, match="adapter"):
        vllm_command("1.5b-lora")
    assert "boxer=/tmp/a" in vllm_command("1.5b-lora", adapter="/tmp/a")


def test_core_plan_covers_each_experiment():
    tags = {s.tag for steps in PLANS["core"].values() for s in steps}
    assert {"baseline", "batching", "caching_full", "capping"} <= tags
    concurrency = {dict(s.kwargs).get("concurrency") for s in PLANS["core"]["3b"]}
    assert {1, 4, 16} <= concurrency


def test_dry_plan_runs(capsys):
    from serving.run_matrix import main
    assert main(["--plan", "core", "--dry-plan"]) == 0
    assert "vllm serve" in capsys.readouterr().out


# ── charts ────────────────────────────────────────────────────────────────────

from bench import charts
from bench.bench import SUMMARY_COLUMNS


def _row(tag, label, c=1, mode="recent", kind="all", p50=200.0, p95=400.0, b50=150.0, stopped=""):
    base = {k: "" for k in SUMMARY_COLUMNS}
    base.update(tag=tag, label=label, context_mode=mode, kind=kind, concurrency=c, fights=10,
                ttft_p50_ms=p50, ttft_p95_ms=p95, ttft_b_p50_ms=b50, decode_tps=50, throughput_tps=80 * c,
                cost_per_fight_usd=0.01, knockdown_recall_described=1.0, invented_per_100=1.0, stopped=stopped)
    return base


def _write_summary(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def test_charts_render_from_recorded_runs(tmp_path):
    rows = [_row("baseline", "3b"), _row("baseline", "3b-nocache", p50=300.0),
            _row("baseline", "3b-awq", p50=150.0),
            _row("batching", "3b", c=4), _row("batching", "3b", c=16),
            _row("caching_full", "3b", mode="full"), _row("caching_full", "3b-nocache", mode="full"),
            _row("capping", "3b", mode="full", kind="long"), _row("capping", "3b", mode="capped", kind="long"),
            _row("routing", "routed:a+b")]
    _write_summary(tmp_path / "summary.csv", rows)
    assert charts.main(["--summary", str(tmp_path / "summary.csv"), "--out", str(tmp_path)]) == 0
    for name in ("ttft_by_backend", "batching", "caching", "quantization", "capping", "routing"):
        assert (tmp_path / f"{name}.png").stat().st_size > 2000, name
    table = (tmp_path / "results_table.md").read_text(encoding="utf-8")
    assert "| 3b |" in table and "Backend" in table


def test_charts_skip_missing_runs_instead_of_inventing(tmp_path, capsys):
    _write_summary(tmp_path / "summary.csv", [_row("baseline", "3b")])
    charts.main(["--summary", str(tmp_path / "summary.csv"), "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert (tmp_path / "ttft_by_backend.png").exists()
    assert not (tmp_path / "batching.png").exists()
    assert "skipped batching" in out


def test_partial_runs_are_not_charted_as_results(tmp_path):
    _write_summary(tmp_path / "summary.csv", [_row("baseline", "groq-70b", stopped="max_requests=150")])
    charts.main(["--summary", str(tmp_path / "summary.csv"), "--out", str(tmp_path)])
    assert not (tmp_path / "ttft_by_backend.png").exists()
    assert "(partial)" in (tmp_path / "results_table.md").read_text(encoding="utf-8")


def test_latest_run_wins():
    rows = charts.load_rows.__wrapped__ if hasattr(charts.load_rows, "__wrapped__") else None
    older, newer = _row("baseline", "3b", p50=500.0), _row("baseline", "3b", p50=100.0)
    for r in (older, newer):
        for k in ("concurrency", "ttft_p50_ms"):
            r[k] = float(r[k])
    assert charts.pick([older, newer], tag="baseline", label="3b")["ttft_p50_ms"] == 100.0


# ── fine-tune data ────────────────────────────────────────────────────────────

from finetune import gen_data, train_lora


def test_accept_filters():
    prompt = "KO rate: 78%"
    assert gen_data.accept('"Alvarez keeps the pressure on."', prompt, "momentum") == "Alvarez keeps the pressure on."
    assert gen_data.accept("", prompt, "x") is None
    assert gen_data.accept("He has a 92% KO rate", prompt, "x") is None            # invented stat
    assert gen_data.accept("Great round so far.", prompt, "knockdown") is None       # knockdown not called
    assert gen_data.accept("DOWN he goes!", prompt, "knockdown") == "DOWN he goes!"
    assert gen_data.accept("word " * 40, prompt, "x") is None                         # too long
    assert gen_data.accept("As an AI I cannot say", prompt, "x") is None


def test_gen_data_dry_run_holds_out_whole_fights(tmp_path):
    cfg = gen_data.GenConfig(examples=40, eval_examples=10, train_fights=6, eval_fights=3,
                             out_dir=str(tmp_path), dry_run=True)
    manifest = gen_data.generate(cfg)
    train = gen_data.__dict__["json"].loads("[" + ",".join((tmp_path / "train.jsonl").read_text().splitlines()) + "]")
    evals = gen_data.__dict__["json"].loads("[" + ",".join((tmp_path / "eval.jsonl").read_text().splitlines()) + "]")
    assert len(train) == 40 and len(evals) == 10
    assert {r["meta"]["fight_id"] for r in train}.isdisjoint({r["meta"]["fight_id"] for r in evals})
    assert all(r["messages"][0]["role"] == "user" and r["messages"][1]["role"] == "assistant" for r in train)
    assert manifest["train"]["knockdown"] > 0                                          # knockdowns are included
    # benchmark fights (seeds 1000-1029) must never be training fights
    assert min(manifest["train"]["fights"]) >= 5000


def test_select_balanced_oversamples_track_b():
    import random
    pool = ([{"prompt": f"a{i}", "track": "A", "event_type": "x", "fight_id": "f"} for i in range(500)] +
            [{"prompt": f"b{i}", "track": "B", "event_type": "excitement_peak", "fight_id": "f"} for i in range(50)] +
            [{"prompt": f"k{i}", "track": "B", "event_type": "knockdown", "fight_id": "f"} for i in range(10)])
    chosen = gen_data.select_balanced(pool, 100, 0.3, random.Random(1))
    assert len(chosen) == 100
    assert sum(c["track"] == "B" for c in chosen) == 30
    assert sum(c["event_type"] == "knockdown" for c in chosen) == 10                  # knockdowns first


def test_recorder_produces_realistic_history():
    prompts = gen_data.record_prompts([5000])
    assert any(p["track"] == "B" and p["event_type"] == "knockdown" for p in prompts)
    a = [p for p in prompts if p["track"] == "A"]
    assert len({p["prompt"] for p in a}) > 10


# ── LoRA tokenization / label masking (fake tokenizer, no torch) ──────────────

class FakeTok:
    eos_token = "<eos>"
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=True):
        return [ord(c) % 200 + 1 for c in "".join(m["content"] for m in msgs)] + [900]   # 900 = assistant marker

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 200 + 1 for c in text.replace("<eos>", "")] + [999]}   # 999 = eos


def test_loss_is_only_on_the_reply():
    msgs = [{"role": "user", "content": "PROMPT"}, {"role": "assistant", "content": "reply"}]
    ex = train_lora.build_example(FakeTok(), msgs)
    n_prompt = len("PROMPT") + 1
    assert ex["labels"][:n_prompt] == [-100] * n_prompt
    assert all(l != -100 for l in ex["labels"][n_prompt:])
    assert ex["labels"][-1] == 999                                                    # model learns to stop
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["attention_mask"])


def test_long_prompts_keep_the_reply_and_the_prompt_tail():
    msgs = [{"role": "user", "content": "x" * 500}, {"role": "assistant", "content": "ok"}]
    ex = train_lora.build_example(FakeTok(), msgs, max_len=100)
    assert len(ex["input_ids"]) == 100
    assert ex["labels"][-3:] != [-100] * 3


def test_collate_pads_labels_with_ignore_index():
    torch = pytest.importorskip("torch")
    b = [{"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1]},
         {"input_ids": [4], "labels": [4], "attention_mask": [1]}]
    out = train_lora.collate(b, pad_id=0)
    assert out["input_ids"].shape == (2, 3)
    assert out["labels"][1].tolist() == [4, -100, -100]
    assert out["attention_mask"][1].tolist() == [1, 0, 0]


# ── voice timing ──────────────────────────────────────────────────────────────

from voice import kokoro_tts
from voice.speak_fight import FakeEngine, summarize, run
from bench.bench import Budget, MeteredLLM, STUB_BACKEND
from bench.stub_llm import StubLLM


def test_time_to_first_audio_is_measured_at_first_chunk():
    engine = FakeEngine(delay_s=0.03)
    t = kokoro_tts.speak("One. Two. Three.", engine, llm_ms=120.0)
    assert t.chunks == 3
    assert 25 <= t.tts_first_audio_ms < t.tts_total_ms
    assert t.e2e_first_audio_ms == pytest.approx(120.0 + t.tts_first_audio_ms)
    assert t.audio_seconds == pytest.approx(1.5)
    assert t.real_time_factor < 1.0


def test_speak_streams_chunks_to_player():
    got = []
    kokoro_tts.speak("A. B.", FakeEngine(), on_audio=got.append)
    assert len(got) == 2


def test_wav_bytes_roundtrip():
    import io, wave
    np = pytest.importorskip("numpy")
    data = kokoro_tts.to_wav_bytes(np.zeros(2400, dtype="float32"))
    with wave.open(io.BytesIO(data)) as w:
        assert (w.getnchannels(), w.getframerate(), w.getnframes()) == (1, 24000, 2400)


def test_split_sentences():
    assert kokoro_tts.split_sentences("He's down! Count it. Wow") == ["He's down!", "Count it.", "Wow"]


def test_voice_run_summarizes_lines():
    llm = MeteredLLM(StubLLM(), STUB_BACKEND, Budget())
    rows = run(llm, FakeEngine(), fights=1, max_lines=10, player=None)
    assert 0 < len(rows) <= 10
    s = summarize(rows)
    assert s["lines"] == len(rows) and s["e2e_first_audio_p50_ms"] is not None


def test_missing_kokoro_gives_a_clear_error():
    try:
        import kokoro  # noqa
        pytest.skip("kokoro is installed")
    except ImportError:
        with pytest.raises(RuntimeError, match="not installed"):
            list(kokoro_tts.KokoroEngine().synthesize("hi"))


# ── deploy manifests / config ─────────────────────────────────────────────────

def test_k8s_manifests_are_valid_and_consistent():
    docs = []
    for name in ("python-service.yaml", "gateway.yaml", "kind-config.yaml"):
        docs += [d for d in yaml.safe_load_all((ROOT / "k8s" / name).read_text(encoding="utf-8")) if d]
    kinds = [d["kind"] for d in docs]
    assert kinds.count("Deployment") == 2 and kinds.count("Service") == 2
    gw = next(d for d in docs if d["kind"] == "Deployment" and d["metadata"]["name"] == "go-gateway")
    env = {e["name"]: e["value"] for e in gw["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert env["PYTHON_SERVICE_URL"] == "http://python-service:8000"
    kind_cfg = next(d for d in docs if d["kind"] == "Cluster")
    node_port = next(d for d in docs if d["kind"] == "Service" and d["metadata"]["name"] == "go-gateway")
    assert kind_cfg["nodes"][0]["extraPortMappings"][0]["containerPort"] == node_port["spec"]["ports"][0]["nodePort"]


def test_compose_passes_backend_settings():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    env = compose["services"]["python-service"]["environment"]
    assert "COMMENTARY_BACKEND" in env and "NEBIUS_API_KEY" in env and "VLLM_BASE_URL" in env


def test_dockerfile_has_curl_for_healthchecks_and_dockerignore_excludes_venv():
    assert "curl" in (ROOT / "Dockerfile").read_text()
    ignore = (ROOT / ".dockerignore").read_text().split()
    assert "venv/" in ignore and ".env" in ignore and "config/" not in ignore


def test_commentary_backend_env_selects_provider(monkeypatch):
    import importlib
    from src.config import llm_config
    monkeypatch.setenv("COMMENTARY_BACKEND", "vllm-3b")
    importlib.reload(llm_config)
    assert llm_config.LLMConfig.PROVIDER == "vllm-3b"
    monkeypatch.delenv("COMMENTARY_BACKEND")
    importlib.reload(llm_config)
    assert llm_config.LLMConfig.PROVIDER == "groq"


def test_notebooks_are_valid():
    for path in ("serving/kaggle_vllm.ipynb", "finetune/train_lora.ipynb"):
        nb = json.loads((ROOT / path).read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4 and nb["cells"]
        # every hosted benchmark cell must carry a spending/quota cap
        for cell in nb["cells"]:
            src = "".join(cell["source"])
            for line in src.splitlines():
                if line.startswith("!python -m bench.bench --backend"):
                    assert "--max-requests" in line or "--max-usd" in line, line


def test_env_file_is_loaded_but_never_overrides(tmp_path, monkeypatch):
    from src.config.backends import load_env, load_backends
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from_file\nNEBIUS_MODEL=some/model\nNEBIUS_BASE_URL=https://x/v1/\n",
                   encoding="utf-8")
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "from_shell")          # already set: must win
    assert load_env(env) is True
    backends = load_backends()
    assert backends["groq-70b"].api_key == "from_shell"
    assert backends["nebius"].model == "some/model"
    assert backends["nebius"].configured
