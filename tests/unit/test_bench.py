import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import csv
import json
import threading
import time

import pytest

from bench import bench
from bench.bench import Budget, BudgetExhausted, MeteredLLM, RunConfig, percentile, run_bench
from bench.fights import make_fight, replay
from bench.quality import invented_stats, score_stats
from bench.stub_llm import StubLLM
from src.config.backends import Backend
from src.core import clock
from src.core.orchestrator import CommentaryOrchestrator
from src.generation.router import FallbackLLM, build_routed_clients


# ── percentile ────────────────────────────────────────────────────────────────

def test_percentile():
    assert percentile([], 50) is None
    assert percentile([5], 95) == 5
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


# ── invented stats ────────────────────────────────────────────────────────────

PROMPT = "Alvarez (60-2-2). KO rate: 78%. Reach: 70.5 inches. [Live] 7/10 landed (70.0% acc)"


def test_grounded_numbers_are_not_flagged():
    assert invented_stats("Alvarez, 78% KO rate, 70.5 inch reach, 60 wins", PROMPT) == []
    assert invented_stats("He landed 7 of 10 at 70% accuracy", PROMPT) == []
    assert invented_stats("a one-two combination from fighter 1", PROMPT) == []


def test_invented_numbers_are_flagged():
    assert invented_stats("He has a 92% KO rate", PROMPT) == ["92"]
    assert invented_stats("record of 61-2-2", PROMPT) == ["61"]


def test_fraction_form_of_a_percentage_is_accepted():
    assert invented_stats("a 0.78 KO rate", PROMPT) == []
    assert invented_stats("a 0.91 KO rate", PROMPT) == ["0.91"]


def test_score_stats_skips_records_without_prompts():
    class R:
        def __init__(self, text, prompt): self.text, self.prompt = text, prompt
    s = score_stats([R("78% KO rate", PROMPT), R("99% KO rate", PROMPT), R("hi", ""), R("", PROMPT)])
    assert (s.generations, s.with_invented, s.invented_numbers) == (2, 1, 1)
    assert s.per_100 == 50.0


# ── budget ────────────────────────────────────────────────────────────────────

def test_budget_request_cap():
    b = Budget(max_requests=2)
    b.acquire(); b.acquire()
    with pytest.raises(BudgetExhausted):
        b.acquire()
    with pytest.raises(BudgetExhausted):     # stays stopped
        b.acquire()
    assert b.stopped == "max_requests=2"


def test_budget_usd_cap_via_metered_client():
    backend = Backend(name="x", base_url="", model="m", price_in=1000.0, price_out=1000.0)
    b = Budget(max_usd=0.001)
    llm = MeteredLLM(StubLLM(), backend, b)
    assert llm.generate("some prompt text") != ""
    assert b.usd > 0
    with pytest.raises(BudgetExhausted):
        llm.generate("again")


def test_budget_pacing():
    b = Budget(rpm=600)                       # one request per 0.1 s
    t0 = time.monotonic()
    for _ in range(4):
        b.acquire()
    assert time.monotonic() - t0 >= 0.25


# ── clock is per thread ───────────────────────────────────────────────────────

def test_virtual_clocks_do_not_leak_between_threads():
    seen = {}

    def run(name, t):
        clock.set_source(lambda: t)
        time.sleep(0.05)
        seen[name] = clock.now()
        clock.reset()

    threads = [threading.Thread(target=run, args=(i, 1000.0 * (i + 1))) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {0: 1000.0, 1: 2000.0, 2: 3000.0, 3: 4000.0}
    assert abs(clock.now() - time.time()) < 5


# ── run_bench end to end (stub backend, no network) ───────────────────────────

def _cfg(tmp_path, **kw):
    base = dict(dry_run=True, fights=3, out_dir=str(tmp_path), warmup=False, tag="t")
    base.update(kw)
    return RunConfig(**base)


def test_dry_run_writes_results(tmp_path):
    res = run_bench(_cfg(tmp_path))
    s = res["summary"]
    assert s["generations"] > 50 and s["error_rate"] == 0.0
    assert s["knockdown_recall_called"] == 1.0
    assert s["cost_per_fight_usd"] > 0
    assert s["usage_estimated_share"] == 1.0           # stub reports no usage
    assert json.loads((tmp_path / "summary.json").read_text())["label"] == "stub"
    with open(tmp_path / "generations.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert {"ttft_ms", "total_ms", "track", "event_type", "invented"} <= set(rows[0])


def test_dropped_knockdowns_show_up_in_recall(tmp_path):
    def make(name):
        return MeteredLLM(StubLLM(drop_knockdowns=True), bench.STUB_BACKEND, Budget())
    res = run_bench(_cfg(tmp_path), make=make)
    assert res["summary"]["knockdown_recall_called"] == 0.0
    assert res["summary"]["error_rate"] > 0           # empty replies counted as errors


def test_run_stops_cleanly_at_request_cap(tmp_path):
    res = run_bench(_cfg(tmp_path, max_requests=10, fights=6))
    assert res["summary"]["stopped"] == "max_requests=10"
    assert res["summary"]["generations"] <= 10
    assert (tmp_path / "summary.json").exists()       # partial results still written


def test_concurrent_run_matches_sequential_counts(tmp_path):
    seq = run_bench(_cfg(tmp_path / "a", concurrency=1))["summary"]
    par = run_bench(_cfg(tmp_path / "b", concurrency=4))["summary"]
    # same fights and same prompts -> same number of generations regardless of concurrency
    assert par["generations"] == seq["generations"]
    assert par["knockdown_recall_called"] == seq["knockdown_recall_called"] == 1.0


def test_latency_is_measured_from_the_backend(tmp_path):
    def make(name):
        return MeteredLLM(StubLLM(delay_s=0.02), bench.STUB_BACKEND, Budget())
    res = run_bench(_cfg(tmp_path, fights=1, kind="short"), make=make)
    assert res["summary"]["ttft_p50_ms"] >= 15


def test_fight_kind_filter():
    assert {len(f.punches) for f in bench.select_fights(10, "long")} == {400}
    assert len(bench.select_fights(4, "short")) == 4


# ── routing ───────────────────────────────────────────────────────────────────

class Named(StubLLM):
    def __init__(self, name, **kw):
        super().__init__(**kw)
        self._name = name

    def get_provider_name(self):
        return self._name


def test_routing_sends_tracks_to_different_backends():
    clients = {"fast": Named("fast"), "smart": Named("smart")}
    a, b = build_routed_clients({"track_a": "smart", "track_b": "fast"}, clients.get)
    fight = make_fight(seed=1, n_punches=120, n_knockdowns=2)
    orch = CommentaryOrchestrator(track_a_llm=a, track_b_llm=b, use_rag=False)
    replay(fight, orch)
    by_track = {}
    for r in orch.generation_log:
        by_track.setdefault(r.track, set()).add(r.backend)
    assert by_track == {"A": {"smart"}, "B": {"fast"}}
    kd = [r for r in orch.generation_log if r.event_type == "knockdown"]
    assert kd and all(r.backend == "fast" for r in kd)


def test_fallback_when_primary_fails():
    class Dead(StubLLM):
        def generate_streaming(self, prompt, **kw):
            return iter(())
        def generate(self, prompt, **kw):
            return ""
    chain = FallbackLLM([Named("primary", ) if False else Dead(), Named("backup")])
    out = "".join(chain.generate_streaming("hello"))
    assert out.strip() != ""
    assert chain.get_provider_name() == "backup"
    assert chain.fallbacks_used == 1


def test_routing_requires_a_target():
    with pytest.raises(ValueError):
        build_routed_clients({"track_a": "x"}, lambda n: StubLLM())


def test_routed_bench_run(tmp_path):
    cfg = _cfg(tmp_path, routing=True, track_a="a", track_b="b")
    res = run_bench(cfg)
    assert res["summary"]["label"] == "routed:a+b"
    assert res["summary"]["generations"] > 0


# ── context modes ─────────────────────────────────────────────────────────────

def _prompts(mode, cap=4):
    seen = []

    class Capture(StubLLM):
        def generate_streaming(self, prompt, **kw):
            seen.append(prompt)
            yield from super().generate_streaming(prompt, **kw)

    fight = make_fight(seed=5, n_punches=200, n_knockdowns=1)
    orch = CommentaryOrchestrator(llm_client=Capture(), use_rag=False,
                                  context_mode=mode, history_cap=cap)
    replay(fight, orch)
    return [p for p in seen if "analyst" in p]


def test_full_mode_grows_and_capped_mode_is_bounded():
    full = _prompts("full")
    capped = _prompts("capped", cap=4)
    recent = _prompts("recent")
    assert len(full[-1]) > len(full[5]) * 1.5            # transcript keeps growing
    assert max(len(p) for p in capped) < max(len(p) for p in full)
    assert capped[-1].count("\n- ") <= 4 + 3            # profiles (2) + cap lines
    assert "COMMENTARY SO FAR" not in "".join(recent)


def test_full_mode_prefix_is_append_only():
    """Each prompt's transcript extends the previous one, so a prefix cache keeps hitting."""
    full = _prompts("full")
    heads = [p.split("--- LIVE SITUATION ---")[0] for p in full]
    for prev, cur in zip(heads, heads[1:]):
        # ignore the closing newlines of the block; the text itself only ever grows
        assert cur.startswith(prev.rstrip())


def test_live_stats_present_without_rag():
    prompts = _prompts("recent")
    assert any("[Live Stats" in p for p in prompts)
