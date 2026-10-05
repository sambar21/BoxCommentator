"""
Charts and the README results table, built from bench/results/summary.csv.

  python -m bench.charts              # all charts + results_table.md into bench/results/
  python -m bench.charts --table      # print the markdown table only

Every number comes from a recorded run; a chart whose runs are missing is
skipped (with a note), never filled in. When the same configuration was run
more than once, the most recent run wins.

Style: light chart surface, blue = slot 1, orange = slot 2 (validated pair),
neutral ink, thin bars, direct value labels, no dual axes.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

RESULTS_DIR = Path(__file__).parent / "results"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"       # categorical slot 1
ORANGE = "#eb6834"     # categorical slot 2
BLUE_LIGHT = "#86b6ef"  # sequential blue 250: same hue, lighter step (p95 next to p50)


# ── data ──────────────────────────────────────────────────────────────────────

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_rows(path: Optional[Path] = None) -> List[dict]:
    path = path or RESULTS_DIR / "summary.csv"
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k not in ("tag", "label", "context_mode", "kind", "stopped", "timestamp"):
                r[k] = _num(v)
    return rows


def pick(rows: List[dict], **where) -> Optional[dict]:
    """Latest row matching all key=value filters (rows are appended in time order)."""
    match = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    return match[-1] if match else None


def complete(r: Optional[dict]) -> bool:
    """A run that was cut short by a budget cap is not a full-sample result."""
    return r is not None and not r.get("stopped")


# ── matplotlib setup ──────────────────────────────────────────────────────────

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": BASELINE, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    })
    return plt


def _finish(plt, fig, title, subtitle, path: Path):
    # Header spacing is set in inches so short and tall figures look the same.
    h = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.0 / h)
    fig.text(0.02, 1 - 0.12 / h, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 1 - 0.46 / h, subtitle, fontsize=9, color=INK_2, va="top")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"wrote {path}")


def _fmt(v, unit=""):
    if v is None:
        return "n/a"
    return f"{v:,.0f}{unit}" if abs(v) >= 10 else f"{v:,.1f}{unit}"


# ── charts ────────────────────────────────────────────────────────────────────

def chart_ttft(rows, out: Path) -> bool:
    """Headline: time-to-first-token p50 and p95 per backend (baseline runs)."""
    base = [r for r in rows if r["tag"] == "baseline" and r["concurrency"] == 1
            and r["context_mode"] == "recent" and complete(r)]
    latest: Dict[str, dict] = {}
    for r in base:
        latest[r["label"]] = r
    items = sorted(latest.values(), key=lambda r: r["ttft_p50_ms"] or 1e9)
    items = [r for r in items if r["ttft_p50_ms"] is not None]
    if not items:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 1.2 + 0.75 * len(items)))
    fig.subplots_adjust(top=0.80, left=0.22, right=0.95, bottom=0.14)
    ys = range(len(items))
    h = 0.34
    p50 = [r["ttft_p50_ms"] for r in items]
    p95 = [r["ttft_p95_ms"] or 0 for r in items]
    ax.barh([y + h / 2 for y in ys], p50, height=h, color=BLUE, label="p50")
    ax.barh([y - h / 2 for y in ys], p95, height=h, color=BLUE_LIGHT, label="p95")
    for y, a, b in zip(ys, p50, p95):
        ax.text(a, y + h / 2, f"  {_fmt(a)} ms", va="center", fontsize=8.5, color=INK_2)
        ax.text(b, y - h / 2, f"  {_fmt(b)} ms", va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in items], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("time to first token (ms), lower is better")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="upper right", fontsize=9)   # fastest rows are on top: room on the right
    ax.set_xlim(0, max(p95 + p50) * 1.18)
    n = items[0]["fights"]
    _finish(plt, fig, "Time to first token by backend",
            f"{int(n)} simulated fights per backend, streaming, one fight at a time", out)
    return True


def chart_batching(rows, out: Path) -> bool:
    """TTFT p95 and throughput vs concurrent fights (two panels, never a dual axis)."""
    pts = []
    for c in (1, 4, 16):
        tag = "baseline" if c == 1 else "batching"
        r = pick(rows, tag=tag, label="3b", concurrency=float(c), context_mode="recent")
        if complete(r):
            pts.append((c, r))
    if len(pts) < 2:
        return False
    plt = _plt()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6))
    fig.subplots_adjust(top=0.78, left=0.08, right=0.97, bottom=0.2, wspace=0.28)
    xs = [str(int(c)) for c, _ in pts]
    for ax, key, title, unit in ((a1, "ttft_p95_ms", "TTFT p95 (ms)", " ms"),
                                 (a2, "throughput_tps", "Throughput (tokens/s)", "")):
        vals = [r[key] for _, r in pts]
        ax.plot(xs, vals, color=BLUE, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2)
        for x, v in zip(xs, vals):
            ax.annotate(_fmt(v, unit), (x, v), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8.5, color=INK_2)
        ax.set_title(title, loc="left", fontsize=10, color=INK_2)
        ax.set_xlabel("concurrent fights")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.grid(axis="x", visible=False)
    _finish(plt, fig, "Batching: more fights at once on one T4",
            "Qwen2.5-3B on vLLM; latency rises as throughput is shared", out)
    return True


def _paired_bars(rows, out, specs, title, subtitle, ylabel):
    """specs: list of (group_label, [(series_label, color, row), ...])."""
    groups = [(g, [(s, c, r) for s, c, r in series if complete(r)]) for g, series in specs]
    groups = [(g, s) for g, s in groups if s]
    if not groups:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.2)
    width = 0.34
    seen = {}
    for gi, (g, series) in enumerate(groups):
        for si, (s, color, r) in enumerate(series):
            x = gi + (si - (len(series) - 1) / 2) * (width + 0.02)
            v = r["ttft_p50_ms"]
            ax.bar(x, v, width=width, color=color, label=s if s not in seen else None)
            seen[s] = True
            ax.text(x, v, _fmt(v), ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g for g, _ in groups], fontsize=9.5, color=INK)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _finish(plt, fig, title, subtitle, out)
    return True


def chart_caching(rows, out: Path) -> bool:
    on_base = pick(rows, tag="baseline", label="3b", concurrency=1.0, context_mode="recent")
    off_base = pick(rows, tag="baseline", label="3b-nocache", concurrency=1.0, context_mode="recent")
    on_full = pick(rows, tag="caching_full", label="3b", context_mode="full")
    off_full = pick(rows, tag="caching_full", label="3b-nocache", context_mode="full")
    return _paired_bars(
        rows, out,
        [("Fixed prompt\n(profiles + last 2 lines)", [("prefix cache on", BLUE, on_base), ("prefix cache off", ORANGE, off_base)]),
         ("Growing prompt\n(whole transcript)", [("prefix cache on", BLUE, on_full), ("prefix cache off", ORANGE, off_full)])],
        "Prefix caching: time to first token (p50)",
        "Qwen2.5-3B on vLLM, same fights and prompts, caching toggled", "TTFT p50 (ms), lower is better")


def chart_quantization(rows, out: Path) -> bool:
    items = []
    for label in ("3b", "3b-awq", "7b-awq"):
        r = pick(rows, tag="baseline", label=label, concurrency=1.0, context_mode="recent")
        if complete(r):
            items.append(r)
    if len(items) < 2:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.2)
    xs = range(len(items))
    vals = [r["ttft_p50_ms"] for r in items]
    ax.bar(xs, vals, width=0.4, color=BLUE)
    for x, r in zip(xs, items):
        kd = r.get("knockdown_recall_described")
        ax.text(x, r["ttft_p50_ms"], f"{_fmt(r['ttft_p50_ms'])} ms", ha="center", va="bottom",
                fontsize=8.5, color=INK_2)
        ax.text(x, -max(vals) * 0.16, f"knockdowns called: {kd:.0%}" if kd is not None else "",
                ha="center", fontsize=8.5, color=INK_2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["label"] for r in items], fontsize=9.5, color=INK)
    ax.set_ylabel("TTFT p50 (ms), lower is better")
    ax.grid(axis="x", visible=False)
    _finish(plt, fig, "Quantization: speed vs. quality",
            "fp16 vs AWQ on a T4; the second line under each bar is the quality check", out)
    return True


def chart_capping(rows, out: Path) -> bool:
    long_runs = [
        ("full transcript\n+ prefix cache", pick(rows, tag="capping", label="3b", context_mode="full", kind="long")),
        ("capped transcript\n+ prefix cache", pick(rows, tag="capping", label="3b", context_mode="capped", kind="long")),
        ("full transcript\nno cache", pick(rows, tag="capping", label="3b-nocache", context_mode="full", kind="long")),
        ("capped transcript\nno cache", pick(rows, tag="capping", label="3b-nocache", context_mode="capped", kind="long")),
    ]
    items = [(n, r) for n, r in long_runs if complete(r)]
    if len(items) < 2:
        return False
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    fig.subplots_adjust(top=0.80, left=0.09, right=0.97, bottom=0.22)
    vals = [r["ttft_p50_ms"] for _, r in items]
    ax.bar(range(len(items)), vals, width=0.4, color=BLUE)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{_fmt(v)} ms", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for n, _ in items], fontsize=9, color=INK)
    ax.set_ylabel("TTFT p50 (ms), lower is better")
    ax.grid(axis="x", visible=False)
    _finish(plt, fig, "Long fights: cap the context or cache it?",
            "400-punch fights, Qwen2.5-3B on vLLM", out)
    return True


def chart_routing(rows, out: Path) -> bool:
    routed = next((r for r in reversed(rows) if r["tag"] == "routing" and str(r["label"]).startswith("routed")
                   and complete(r)), None)
    singles = [r for r in rows if r["tag"] == "baseline" and r["concurrency"] == 1.0
               and r["context_mode"] == "recent" and complete(r) and r["ttft_b_p50_ms"] is not None]
    if routed is None or not singles:
        return False
    best_single = min(singles, key=lambda r: r["ttft_b_p50_ms"])
    return _paired_bars_b(rows, out, routed, best_single)


def _paired_bars_b(rows, out, routed, single) -> bool:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    fig.subplots_adjust(top=0.80, left=0.12, right=0.97, bottom=0.2)
    items = [(f"fastest single model\n({single['label']})", single["ttft_b_p50_ms"], BLUE_LIGHT),
             (f"routed\n({routed['label'].replace('routed:', '')})", routed["ttft_b_p50_ms"], BLUE)]
    for i, (_, v, c) in enumerate(items):
        ax.bar(i, v, width=0.4, color=c)
        ax.text(i, v, f"{_fmt(v)} ms", ha="center", va="bottom", fontsize=9, color=INK_2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([n for n, _, _ in items], fontsize=9, color=INK)
    ax.set_ylabel("knockdown call TTFT p50 (ms)")
    ax.grid(axis="x", visible=False)
    _finish(plt, fig, "Routing: time to call a knockdown",
            f"routed knockdown recall {routed['knockdown_recall_described']:.0%} "
            f"vs {single['knockdown_recall_described']:.0%} for the single model", out)
    return True


CHARTS = {
    "ttft_by_backend": chart_ttft,
    "batching": chart_batching,
    "caching": chart_caching,
    "quantization": chart_quantization,
    "capping": chart_capping,
    "routing": chart_routing,
}


# ── results table ─────────────────────────────────────────────────────────────

TABLE_COLUMNS = [
    ("label", "Backend"), ("ttft_p50_ms", "TTFT p50 (ms)"), ("ttft_p95_ms", "TTFT p95 (ms)"),
    ("ttft_b_p50_ms", "Knockdown TTFT p50 (ms)"), ("decode_tps", "Tokens/s"),
    ("cost_per_fight_usd", "$/fight"), ("knockdown_recall_described", "Knockdowns called"),
    ("invented_per_100", "Invented stats /100"),
]


def results_table(rows: List[dict]) -> str:
    base = [r for r in rows if r["tag"] == "baseline" and r["concurrency"] == 1
            and r["context_mode"] == "recent"]
    latest: Dict[str, dict] = {}
    for r in base:
        latest[r["label"]] = r
    items = sorted(latest.values(), key=lambda r: r["ttft_p50_ms"] or 1e9)
    if not items:
        return "_No benchmark runs recorded yet._\n"
    head = "| " + " | ".join(h for _, h in TABLE_COLUMNS) + " |"
    sep = "|" + "|".join("---" for _ in TABLE_COLUMNS) + "|"
    lines = [head, sep]
    for r in items:
        cells = []
        for key, _ in TABLE_COLUMNS:
            v = r.get(key)
            if key == "label":
                cells.append(f"{v}" + (" *(partial)*" if r.get("stopped") else ""))
            elif v is None:
                cells.append("n/a")
            elif key == "knockdown_recall_described":
                cells.append(f"{v:.0%}")
            elif key == "cost_per_fight_usd":
                cells.append(f"{v:.4f}")
            else:
                cells.append(_fmt(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--table", action="store_true", help="print the markdown results table and exit")
    p.add_argument("--summary", help="summary.csv path (default bench/results/summary.csv)")
    p.add_argument("--out", help="output directory (default bench/results)")
    a = p.parse_args(argv)
    rows = load_rows(Path(a.summary) if a.summary else None)
    if a.table:
        print(results_table(rows))
        return 0
    if not rows:
        print("no runs in bench/results/summary.csv yet; nothing to chart", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in CHARTS.items():
        if not fn(rows, out / f"{name}.png"):
            print(f"skipped {name}: the runs it needs are not recorded (or were cut short)")
    (out / "results_table.md").write_text(results_table(rows), encoding="utf-8")
    print(f"wrote {out / 'results_table.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
