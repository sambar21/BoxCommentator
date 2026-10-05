"""
Benchmark runner: replay simulated fights against a backend (or a routed pair)
with streaming responses and log latency, throughput, cost and quality.

Examples
  python -m bench.bench --list
  python -m bench.bench --dry-run --fights 3                       # no network, validates the harness
  python -m bench.bench --backend groq-70b --fights 3 --max-requests 120 --rpm 25 --tag baseline
  python -m bench.bench --backend vllm-3b --fights 10 --concurrency 4 --tag batching
  python -m bench.bench --routing --track-a groq-70b --track-b vllm-3b --tag routing

Safety: --max-requests and --max-usd stop a run cleanly (partial results are
still written). --rpm paces requests for rate-limited free tiers.

Timing note: fight logic runs on a virtual clock (instant); only LLM calls take
real time, and that is what is measured.
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from bench.fights import Fight, make_suite, replay
from bench.quality import aggregate, knockdown_recall, merge_stat_scores, score_stats, invented_stats
from bench.stub_llm import StubLLM
from src.config.backends import Backend, load_backends, load_routing
from src.core.orchestrator import CommentaryOrchestrator
from src.generation.llm_interface.base_client import BaseLLMClient
from src.generation.llm_interface.openai_compat_client import OpenAICompatClient
from src.generation.router import build_routed_clients

RESULTS_DIR = Path(__file__).parent / "results"

FIGHT_KINDS = {"short": 60, "standard": 150, "long": 400}

SUMMARY_COLUMNS = [
    "tag", "label", "context_mode", "kind", "concurrency", "fights", "repeats", "generations",
    "error_rate", "ttft_p50_ms", "ttft_p95_ms", "ttft_b_p50_ms", "ttft_b_p95_ms",
    "total_p50_ms", "total_p95_ms", "decode_tps", "throughput_tps", "wall_s",
    "cost_per_fight_usd", "usage_estimated_share", "knockdown_recall_called",
    "knockdown_recall_described", "invented_per_100", "cache_hit_rate_server",
    "cache_hit_rate_usage", "stopped", "timestamp",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile (q in 0..100); None for no data."""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q / 100.0
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def _round(x, nd=1):
    return None if x is None else round(x, nd)


# ── budget + metering ─────────────────────────────────────────────────────────

class BudgetExhausted(Exception):
    pass


class Budget:
    """Shared across all clients in a run: request cap, dollar cap, pacing."""

    def __init__(self, max_requests: Optional[int] = None, max_usd: Optional[float] = None,
                 rpm: Optional[float] = None):
        self.max_requests = max_requests
        self.max_usd = max_usd
        self.rpm = rpm
        self.requests = 0
        self.usd = 0.0
        self.stopped: Optional[str] = None
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def acquire(self):
        sleep_for = 0.0
        with self._lock:
            if self.stopped:
                raise BudgetExhausted(self.stopped)
            if self.max_requests is not None and self.requests >= self.max_requests:
                self.stopped = f"max_requests={self.max_requests}"
                raise BudgetExhausted(self.stopped)
            if self.max_usd is not None and self.usd >= self.max_usd:
                self.stopped = f"max_usd={self.max_usd}"
                raise BudgetExhausted(self.stopped)
            self.requests += 1
            if self.rpm:
                now = time.monotonic()
                slot = max(now, self._next_slot)
                self._next_slot = slot + 60.0 / self.rpm
                sleep_for = slot - now
        if sleep_for > 0:
            time.sleep(sleep_for)

    def charge(self, usd: float):
        with self._lock:
            self.usd += usd


class MeteredLLM(BaseLLMClient):
    """Wraps a client: enforces the shared Budget and tracks estimated spend."""

    def __init__(self, inner: BaseLLMClient, backend: Backend, budget: Budget):
        self.inner = inner
        self.backend = backend
        self.budget = budget

    def _charge(self, prompt: str, out_chars: int):
        pt = getattr(self.inner, "last_prompt_tokens", None) or len(prompt) / 4
        ct = getattr(self.inner, "last_completion_tokens", None) or out_chars / 4
        self.budget.charge((pt * self.backend.price_in + ct * self.backend.price_out) / 1e6)

    def generate(self, prompt: str, **kwargs) -> str:
        self.budget.acquire()
        out = self.inner.generate(prompt, **kwargs)
        self._charge(prompt, len(out))
        return out

    def generate_streaming(self, prompt: str, **kwargs):
        self.budget.acquire()
        out_chars = 0
        try:
            for chunk in self.inner.generate_streaming(prompt, **kwargs):
                out_chars += len(chunk)
                yield chunk
        finally:
            self._charge(prompt, out_chars)

    def health_check(self) -> bool:
        return self.inner.health_check()

    def get_provider_name(self) -> str:
        return self.inner.get_provider_name()

    def warmup(self):
        """One tiny request outside the budget so cold-start cost is not measured."""
        try:
            self.inner.generate("Reply with the single word: ready", max_tokens=5)
        except Exception:
            pass

    def __getattr__(self, name):          # last_prompt_tokens, last_cached_tokens, ...
        if name in ("inner", "backend", "budget"):
            raise AttributeError(name)
        return getattr(self.inner, name)


# ── run config / factories ────────────────────────────────────────────────────

@dataclass
class RunConfig:
    backend: Optional[str] = None
    routing: bool = False
    track_a: Optional[str] = None
    track_b: Optional[str] = None
    fallback: List[str] = field(default_factory=list)
    label: Optional[str] = None
    tag: str = "adhoc"
    fights: int = 10
    kind: str = "all"
    repeats: int = 1
    concurrency: int = 1
    context_mode: str = "recent"
    history_cap: int = 8
    max_tokens_a: Optional[int] = None
    max_tokens_b: Optional[int] = None
    max_requests: Optional[int] = None
    max_usd: Optional[float] = None
    rpm: Optional[float] = None
    dry_run: bool = False
    warmup: bool = True
    scrape_metrics: bool = True
    out_dir: Optional[str] = None


STUB_BACKEND = Backend(name="stub", base_url="", model="stub", price_in=0.5, price_out=1.5)


def select_fights(n: int, kind: str = "all") -> List[Fight]:
    suite = make_suite(30)
    if kind != "all":
        size = FIGHT_KINDS[kind]
        suite = [f for f in suite if len(f.punches) == size]
    return suite[:n]


def make_client_factory(cfg: RunConfig, backends: Dict[str, Backend], budget: Budget):
    """name -> MeteredLLM. Dry runs use the stub so nothing touches the network."""
    cache: Dict[str, MeteredLLM] = {}

    def make(name: str) -> MeteredLLM:
        if name in cache:
            return cache[name]
        if cfg.dry_run:
            client = MeteredLLM(StubLLM(), STUB_BACKEND, budget)
        else:
            backend = backends.get(name)
            if backend is None:
                raise SystemExit(f"unknown backend '{name}'. Known: {', '.join(backends)}")
            if not backend.configured:
                raise SystemExit(f"backend '{name}' has no base_url/model; set its env vars "
                                 f"(see config/backends.yaml)")
            client = MeteredLLM(OpenAICompatClient(backend), backend, budget)
        cache[name] = client
        return client

    return make


# ── scraping the vLLM server ──────────────────────────────────────────────────

def _server_root(base_url: str) -> str:
    root = base_url.rstrip("/")
    return root[:-3] if root.endswith("/v1") else root


def scrape_prefix_cache(base_url: str) -> Optional[dict]:
    """Best-effort read of vLLM's Prometheus prefix-cache counters. Names vary by version."""
    try:
        with urllib.request.urlopen(_server_root(base_url) + "/metrics", timeout=5) as r:
            text = r.read().decode()
    except Exception:
        return None
    hits = queries = gauge = None
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, _, value = line.rpartition(" ")
        base = name.split("{")[0]
        try:
            v = float(value)
        except ValueError:
            continue
        if base.endswith("prefix_cache_hits_total") or base.endswith("prefix_cache_hits"):
            hits = (hits or 0.0) + v
        elif base.endswith("prefix_cache_queries_total") or base.endswith("prefix_cache_queries"):
            queries = (queries or 0.0) + v
        elif base.endswith("prefix_cache_hit_rate"):
            gauge = v
    if hits is None and queries is None and gauge is None:
        return None
    return {"hits": hits, "queries": queries, "gauge": gauge}


def server_version(base_url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(_server_root(base_url) + "/version", timeout=5) as r:
            return json.loads(r.read().decode()).get("version")
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=Path(__file__).parent).decode().strip()
    except Exception:
        return None


# ── the run ───────────────────────────────────────────────────────────────────

@dataclass
class FightResult:
    fight: Fight
    rep: int
    log: list
    stopped: Optional[str] = None


def _run_fight(fight: Fight, rep: int, cfg: RunConfig, make: Callable[[str], MeteredLLM]) -> FightResult:
    if cfg.routing:
        a_name = cfg.track_a or cfg.backend
        b_name = cfg.track_b or cfg.backend
        routing = {"track_a": a_name, "track_b": b_name, "fallback": cfg.fallback}
        llm_a, llm_b = build_routed_clients(routing, make)
        orch = CommentaryOrchestrator(
            track_a_llm=llm_a, track_b_llm=llm_b, use_rag=False,
            context_mode=cfg.context_mode, history_cap=cfg.history_cap, keep_prompts=True)
    else:
        orch = CommentaryOrchestrator(
            llm_client=make(cfg.backend or "stub"), use_rag=False,
            context_mode=cfg.context_mode, history_cap=cfg.history_cap, keep_prompts=True)
    if cfg.max_tokens_a:
        orch.track_a_generator.max_tokens = cfg.max_tokens_a
    if cfg.max_tokens_b:
        orch.track_b_generator.max_tokens = cfg.max_tokens_b

    stopped = None
    try:
        replay(fight, orch)
    except BudgetExhausted as e:
        stopped = str(e)
    return FightResult(fight=fight, rep=rep, log=orch.generation_log, stopped=stopped)


def run_bench(cfg: RunConfig, make: Optional[Callable] = None,
              backends: Optional[Dict[str, Backend]] = None) -> dict:
    """Run the benchmark and return {'summary': ..., 'rows': [...], 'meta': ..., 'dir': Path}."""
    backends = backends if backends is not None else ({} if cfg.dry_run else load_backends())
    if cfg.dry_run and not cfg.backend and not cfg.routing:
        cfg.backend = "stub"
    if not cfg.routing and not cfg.backend:
        raise SystemExit("pick --backend NAME, --routing, or --dry-run")
    budget = Budget(cfg.max_requests, cfg.max_usd, cfg.rpm)
    make = make or make_client_factory(cfg, backends, budget)

    names = ([cfg.track_a or cfg.backend, cfg.track_b or cfg.backend] + list(cfg.fallback)
             if cfg.routing else [cfg.backend])
    names = [n for n in dict.fromkeys(names) if n]
    label = cfg.label or (f"routed:{cfg.track_a or cfg.backend}+{cfg.track_b or cfg.backend}"
                          if cfg.routing else cfg.backend)

    fights = select_fights(cfg.fights, cfg.kind)
    if not fights:
        raise SystemExit(f"no fights match kind={cfg.kind}")

    if cfg.warmup and not cfg.dry_run:
        for n in names:
            make(n).warmup()

    # server-side prefix-cache counters (self-hosted only)
    scrape_url = None
    for n in names:
        b = backends.get(n)
        if cfg.scrape_metrics and b and b.self_hosted and b.base_url:
            scrape_url = b.base_url
            break
    cache_before = scrape_prefix_cache(scrape_url) if scrape_url else None

    results: List[FightResult] = []
    wall_total = 0.0
    for rep in range(cfg.repeats):
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, cfg.concurrency)) as pool:
            futures = [pool.submit(_run_fight, f, rep, cfg, make) for f in fights]
            batch = [fut.result() for fut in futures]
        wall_total += time.perf_counter() - t0
        results.extend(batch)
        if budget.stopped:
            break

    cache_after = scrape_prefix_cache(scrape_url) if scrape_url else None

    rows = _build_rows(results, cfg, label)
    summary = summarize(cfg, label, results, rows, wall_total, backends,
                        budget, cache_before, cache_after)
    meta = {
        "config": asdict(cfg),
        "label": label,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "vllm_version": server_version(scrape_url) if scrape_url else None,
        "backends": {n: {"model": backends[n].model, "base_url": backends[n].base_url,
                         "price_in": backends[n].price_in, "price_out": backends[n].price_out,
                         "gpu_hour_usd": backends[n].gpu_hour_usd}
                     for n in names if n in backends},
        "requests_made": budget.requests,
        "estimated_spend_usd": round(budget.usd, 6),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out = _write_outputs(cfg, label, rows, summary, meta)
    return {"summary": summary, "rows": rows, "meta": meta, "dir": out}


def _build_rows(results: List[FightResult], cfg: RunConfig, label: str) -> List[dict]:
    rows = []
    for r in results:
        for rec in r.log:
            ok = bool(rec.text) and rec.ttft_ms is not None
            invented = invented_stats(rec.text, rec.prompt) if (rec.text and rec.prompt) else []
            rows.append({
                "tag": cfg.tag, "label": label, "rep": r.rep, "fight_id": r.fight.fight_id,
                "track": rec.track, "backend": rec.backend, "event_type": rec.event_type,
                "punch_index": rec.punch_index, "ok": int(ok),
                "ttft_ms": _round(rec.ttft_ms, 2), "total_ms": _round(rec.total_ms, 2),
                "n_chunks": rec.n_chunks, "n_tokens": rec.n_tokens,
                "prompt_tokens": rec.prompt_tokens, "cached_tokens": rec.cached_tokens,
                "prompt_chars": rec.prompt_chars, "invented": ";".join(invented),
                "text": rec.text,
            })
    return rows


def _generation_cost(row: dict, backend: Optional[Backend]) -> (float, bool):
    """USD for one generation (API pricing) and whether token counts were estimated."""
    if backend is None:
        return 0.0, False
    estimated = row["prompt_tokens"] is None or row["n_tokens"] is None
    pt = row["prompt_tokens"] if row["prompt_tokens"] is not None else row["prompt_chars"] / 4
    if row["n_tokens"] is not None:
        ct = row["n_tokens"]
    elif row["n_chunks"]:
        ct = row["n_chunks"]
    else:
        ct = len(row["text"]) / 4
    return (pt * backend.price_in + ct * backend.price_out) / 1e6, estimated


def summarize(cfg, label, results, rows, wall_s, backends, budget,
              cache_before=None, cache_after=None) -> dict:
    good = [r for r in rows if r["ok"]]
    ttft = [r["ttft_ms"] for r in good]
    ttft_b = [r["ttft_ms"] for r in good if r["track"] == "B"]
    total = [r["total_ms"] for r in good]

    decode = []
    tokens_total = 0
    for r in good:
        n = r["n_tokens"] if r["n_tokens"] is not None else r["n_chunks"]
        tokens_total += n or 0
        gen_ms = r["total_ms"] - r["ttft_ms"]
        if n and n >= 8 and gen_ms > 0:       # very short replies make tokens/sec meaningless
            decode.append(n / (gen_ms / 1000.0))

    fights_done = len(results)
    api_cost = 0.0
    estimated = 0
    self_hosted_used = set()
    for r in rows:
        backend = backends.get(r["backend"], STUB_BACKEND if cfg.dry_run else None)
        if backend is not None and backend.self_hosted:
            self_hosted_used.add(backend.name)
        usd, est = _generation_cost(r, backend)
        api_cost += usd
        estimated += int(est)
    gpu_cost = sum(backends[n].gpu_hour_usd for n in self_hosted_used) * wall_s / 3600.0
    cost_per_fight = (api_cost + gpu_cost) / fights_done if fights_done else None

    kd = aggregate([knockdown_recall(r.fight, r.log) for r in results])
    qs = merge_stat_scores([score_stats(r.log) for r in results])

    server_rate = None
    if cache_before and cache_after:
        if cache_after.get("queries") is not None and cache_before.get("queries") is not None:
            dq = cache_after["queries"] - cache_before["queries"]
            dh = (cache_after.get("hits") or 0) - (cache_before.get("hits") or 0)
            server_rate = dh / dq if dq > 0 else None
        elif cache_after.get("gauge") is not None:
            server_rate = cache_after["gauge"]
    cached = sum(r["cached_tokens"] or 0 for r in good)
    prompt = sum(r["prompt_tokens"] or 0 for r in good if r["cached_tokens"] is not None)
    usage_rate = cached / prompt if prompt else None

    return {
        "tag": cfg.tag, "label": label, "context_mode": cfg.context_mode, "kind": cfg.kind,
        "concurrency": cfg.concurrency, "fights": fights_done, "repeats": cfg.repeats,
        "generations": len(rows),
        "error_rate": _round((len(rows) - len(good)) / len(rows), 4) if rows else None,
        "ttft_p50_ms": _round(percentile(ttft, 50)), "ttft_p95_ms": _round(percentile(ttft, 95)),
        "ttft_b_p50_ms": _round(percentile(ttft_b, 50)), "ttft_b_p95_ms": _round(percentile(ttft_b, 95)),
        "total_p50_ms": _round(percentile(total, 50)), "total_p95_ms": _round(percentile(total, 95)),
        "decode_tps": _round(sum(decode) / len(decode)) if decode else None,
        "throughput_tps": _round(tokens_total / wall_s) if wall_s > 0 else None,
        "wall_s": _round(wall_s, 2),
        "cost_per_fight_usd": _round(cost_per_fight, 6),
        "usage_estimated_share": _round(estimated / len(rows), 3) if rows else None,
        "knockdown_recall_called": _round(kd.recall_called, 4),
        "knockdown_recall_described": _round(kd.recall_described, 4),
        "knockdowns": kd.total,
        "invented_per_100": _round(qs.per_100, 2),
        "cache_hit_rate_server": _round(server_rate, 4),
        "cache_hit_rate_usage": _round(usage_rate, 4),
        "stopped": budget.stopped,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def _write_outputs(cfg: RunConfig, label: str, rows, summary, meta) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_+." else "_" for c in label)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(cfg.out_dir) if cfg.out_dir else RESULTS_DIR / cfg.tag / f"{safe}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    if rows:
        with open(out / "generations.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # running table of every run, used by charts.py
    table = RESULTS_DIR / "summary.csv"
    if not cfg.out_dir:
        table.parent.mkdir(parents=True, exist_ok=True)
        new = not table.exists()
        with open(table, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(summary)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def print_summary(summary: dict):
    keys = ["label", "tag", "fights", "generations", "error_rate", "ttft_p50_ms", "ttft_p95_ms",
            "ttft_b_p50_ms", "total_p95_ms", "decode_tps", "throughput_tps", "wall_s",
            "cost_per_fight_usd", "knockdown_recall_called", "knockdown_recall_described",
            "invented_per_100", "cache_hit_rate_server", "cache_hit_rate_usage", "stopped"]
    for k in keys:
        print(f"  {k:28} {summary.get(k)}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark commentary backends on simulated fights.")
    p.add_argument("--list", action="store_true", help="list configured backends and exit")
    p.add_argument("--backend")
    p.add_argument("--routing", action="store_true")
    p.add_argument("--track-a")
    p.add_argument("--track-b")
    p.add_argument("--fallback", nargs="*", default=None)
    p.add_argument("--label")
    p.add_argument("--tag", default="adhoc")
    p.add_argument("--fights", type=int, default=10)
    p.add_argument("--kind", choices=["all"] + list(FIGHT_KINDS), default="all")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--context-mode", choices=["recent", "full", "capped"], default="recent")
    p.add_argument("--history-cap", type=int, default=8)
    p.add_argument("--max-tokens-a", type=int)
    p.add_argument("--max-tokens-b", type=int)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--max-usd", type=float)
    p.add_argument("--rpm", type=float)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-scrape", action="store_true")
    p.add_argument("--out-dir")
    return p.parse_args(argv)


def config_from_args(a: argparse.Namespace) -> RunConfig:
    routing_cfg = {} if a.dry_run else load_routing()
    fallback = a.fallback if a.fallback is not None else (
        [routing_cfg["fallback"]] if isinstance(routing_cfg.get("fallback"), str)
        else list(routing_cfg.get("fallback") or []))
    return RunConfig(
        backend=a.backend or (routing_cfg.get("default") if not a.dry_run and not a.routing else None),
        routing=a.routing,
        track_a=a.track_a or (routing_cfg.get("track_a") if a.routing else None),
        track_b=a.track_b or (routing_cfg.get("track_b") if a.routing else None),
        fallback=fallback if a.routing else [],
        label=a.label, tag=a.tag, fights=a.fights, kind=a.kind, repeats=a.repeats,
        concurrency=a.concurrency, context_mode=a.context_mode, history_cap=a.history_cap,
        max_tokens_a=a.max_tokens_a, max_tokens_b=a.max_tokens_b,
        max_requests=a.max_requests, max_usd=a.max_usd, rpm=a.rpm,
        dry_run=a.dry_run, warmup=not a.no_warmup, scrape_metrics=not a.no_scrape,
        out_dir=a.out_dir,
    )


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        for name, b in load_backends().items():
            kind = "self-hosted" if b.self_hosted else "hosted"
            print(f"{name:18} {'ready' if b.configured else 'NOT CONFIGURED':15} {kind:12} {b.model or '-'}")
        return 0
    cfg = config_from_args(args)
    if not cfg.dry_run and not (cfg.max_requests or cfg.max_usd):
        print("warning: no --max-requests/--max-usd set; this run has no spending or quota cap",
              file=sys.stderr)
    result = run_bench(cfg)
    print(f"\n{result['summary']['label']}  ->  {result['dir']}")
    print_summary(result["summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
