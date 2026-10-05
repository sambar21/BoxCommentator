"""
Build the LoRA training set by distilling a large "teacher" model.

  python -m finetune.gen_data --dry-run                       # no network, checks the pipeline
  python -m finetune.gen_data --teacher nebius --max-usd 0.50 # real run (needs NEBIUS_* in .env)

How it works (and why it is cheap)
  1. Replay simulated fights with a free placeholder LLM, only to *record the exact prompts*
     the app would send (Track A analysis prompts and Track B urgent prompts).
  2. Pick a balanced set of those prompts (Track B / knockdown prompts are rare, so they are
     oversampled) and ask the teacher for the commentary line for each one.
  3. Keep only answers that pass quality filters (no invented stats, knockdowns actually
     called, short). Whole fights are held out for the eval split, so eval prompts are unseen.

Fights use seeds from 5000 up; the benchmark suite uses 1000-1029, so no benchmark fight
appears in training data.
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from bench.bench import Budget, BudgetExhausted, MeteredLLM, STUB_BACKEND
from bench.fights import make_fight, replay
from bench.quality import KNOCKDOWN_WORDS, invented_stats
from bench.stub_llm import StubLLM
from src.core.orchestrator import CommentaryOrchestrator
from src.generation.llm_interface.base_client import BaseLLMClient

DATA_DIR = Path(__file__).parent / "data"
TRAIN_SEED_BASE = 5000
MAX_WORDS = 30
TRACK_PARAMS = {"A": {"max_tokens": 40, "temperature": 0.7},
                "B": {"max_tokens": 50, "temperature": 0.9}}

_EVENT_LINE = re.compile(r"^(?:Event|URGENT EVENT): (.+)$", re.MULTILINE)


class RecorderLLM(BaseLLMClient):
    """Free stand-in that returns a short line derived from the event text.
    Gives recorded prompts a realistic, varied 'recent commentary' history."""

    @staticmethod
    def _reply(prompt: str) -> str:
        m = _EVENT_LINE.search(prompt)
        words = (m.group(1) if m else "Steady action").split()[:12]
        return " ".join(words)

    def generate(self, prompt: str, **kw) -> str:
        return self._reply(prompt)

    def generate_streaming(self, prompt: str, **kw):
        yield self._reply(prompt)

    def health_check(self) -> bool:
        return True

    def get_provider_name(self) -> str:
        return "recorder"


@dataclass
class GenConfig:
    teacher: str = "nebius-teacher"
    examples: int = 300
    eval_examples: int = 60
    b_share: float = 0.30            # share of Track B (urgent) examples
    train_fights: int = 24
    eval_fights: int = 6
    oversample: float = 1.4          # ask the teacher for extra to survive filtering
    max_usd: Optional[float] = None
    max_requests: Optional[int] = None
    rpm: Optional[float] = None
    workers: int = 4
    seed: int = 7
    out_dir: str = str(DATA_DIR)
    dry_run: bool = False


# ── step 1: record prompts ────────────────────────────────────────────────────

def record_prompts(seeds: List[int]) -> List[dict]:
    """Replay fights with the placeholder LLM and keep every prompt the app built."""
    out = []
    for i, seed in enumerate(seeds):
        n_punches, n_kd = [(60, 1), (150, 2), (400, 3)][i % 3]
        fight = make_fight(seed, n_punches=n_punches, n_knockdowns=n_kd)
        orch = CommentaryOrchestrator(llm_client=RecorderLLM(), use_rag=False, keep_prompts=True)
        replay(fight, orch)
        for rec in orch.generation_log:
            out.append({"fight_id": fight.fight_id, "track": rec.track, "event_type": rec.event_type,
                        "prompt": rec.prompt})
    return out


# ── step 2: choose a balanced set ─────────────────────────────────────────────

def select_balanced(pool: List[dict], n: int, b_share: float, rng: random.Random) -> List[dict]:
    """n prompts with about b_share Track B, de-duplicated, spread over fights and event types."""
    seen, unique = set(), []
    for item in pool:
        if item["prompt"] not in seen:
            seen.add(item["prompt"])
            unique.append(item)
    a = [x for x in unique if x["track"] == "A"]
    b = [x for x in unique if x["track"] == "B"]
    rng.shuffle(a)
    rng.shuffle(b)
    # knockdowns are the most important Track B case: make sure they are in
    b.sort(key=lambda x: x["event_type"] != "knockdown")
    n_b = min(len(b), round(n * b_share))
    n_a = min(len(a), n - n_b)
    return a[:n_a] + b[:n_b]


# ── step 3: teacher + filters ─────────────────────────────────────────────────

def accept(text: str, prompt: str, event_type: str) -> Optional[str]:
    """Return the cleaned text if it is a usable training target, else None."""
    text = (text or "").strip().strip('"').strip()
    if not text:
        return None
    if len(text.split()) > MAX_WORDS:
        return None
    if invented_stats(text, prompt):
        return None
    if event_type == "knockdown" and not KNOCKDOWN_WORDS.search(text):
        return None
    if re.search(r"as an ai|i cannot|i can't", text, re.IGNORECASE):
        return None
    return text


def ask_teacher(teacher: BaseLLMClient, item: dict) -> Optional[dict]:
    params = TRACK_PARAMS[item["track"]]
    try:
        text = teacher.generate(item["prompt"], **params)
    except BudgetExhausted:
        raise
    except Exception:
        return None
    clean = accept(text, item["prompt"], item["event_type"])
    if clean is None:
        return None
    return {"messages": [{"role": "user", "content": item["prompt"]},
                         {"role": "assistant", "content": clean}],
            "meta": {"fight_id": item["fight_id"], "track": item["track"],
                     "event_type": item["event_type"]}}


def distill(items: List[dict], teacher: BaseLLMClient, target: int, workers: int) -> (List[dict], int):
    """Ask the teacher for each item until `target` examples are accepted. Returns (examples, asked)."""
    examples, asked = [], 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        # submit in small batches so we stop asking as soon as the target is reached
        i = 0
        while i < len(items) and len(examples) < target:
            batch = items[i:i + max(workers, 1) * 4]
            i += len(batch)
            try:
                results = list(pool.map(lambda it: ask_teacher(teacher, it), batch))
            except BudgetExhausted:
                break
            asked += len(batch)
            examples.extend(r for r in results if r)
    return examples[:target], asked


def make_teacher(cfg: GenConfig, budget: Budget) -> BaseLLMClient:
    if cfg.dry_run:
        return MeteredLLM(StubLLM(), STUB_BACKEND, budget)
    from src.config.backends import get_backend
    from src.generation.llm_interface.openai_compat_client import OpenAICompatClient
    backend = get_backend(cfg.teacher)
    if backend is None or not backend.configured:
        raise SystemExit(f"teacher backend '{cfg.teacher}' is not configured (see config/backends.yaml "
                         f"and set its environment variables in .env)")
    return MeteredLLM(OpenAICompatClient(backend, timeout=60), backend, budget)


def generate(cfg: GenConfig, teacher: Optional[BaseLLMClient] = None) -> dict:
    rng = random.Random(cfg.seed)
    budget = Budget(cfg.max_requests, cfg.max_usd, cfg.rpm)
    teacher = teacher or make_teacher(cfg, budget)

    seeds = [TRAIN_SEED_BASE + i for i in range(cfg.train_fights + cfg.eval_fights)]
    train_seeds, eval_seeds = seeds[:cfg.train_fights], seeds[cfg.train_fights:]

    train_pool = select_balanced(record_prompts(train_seeds),
                                 int(cfg.examples * cfg.oversample), cfg.b_share, rng)
    eval_pool = select_balanced(record_prompts(eval_seeds),
                                int(cfg.eval_examples * cfg.oversample), cfg.b_share, rng)
    # sanity: whole fights are held out
    assert not ({x["fight_id"] for x in train_pool} & {x["fight_id"] for x in eval_pool})

    # shuffle so a mid-run budget stop still leaves a balanced set
    rng.shuffle(train_pool)
    rng.shuffle(eval_pool)
    train, asked_t = distill(train_pool, teacher, cfg.examples, cfg.workers)
    evals, asked_e = distill(eval_pool, teacher, cfg.eval_examples, cfg.workers)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train.jsonl", train), ("eval.jsonl", evals)):
        with open(out / name, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def counts(rows):
        c = {"A": 0, "B": 0, "knockdown": 0}
        for r in rows:
            c[r["meta"]["track"]] += 1
            c["knockdown"] += r["meta"]["event_type"] == "knockdown"
        return c

    manifest = {
        "teacher": cfg.teacher if not cfg.dry_run else "stub (dry run)",
        "train": {"examples": len(train), "asked": asked_t, "fights": train_seeds, **counts(train)},
        "eval": {"examples": len(evals), "asked": asked_e, "fights": eval_seeds, **counts(evals)},
        "requests": budget.requests, "estimated_spend_usd": round(budget.usd, 4),
        "stopped": budget.stopped, "config": asdict(cfg),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="nebius-teacher")
    p.add_argument("--examples", type=int, default=300)
    p.add_argument("--eval-examples", type=int, default=60)
    p.add_argument("--max-usd", type=float)
    p.add_argument("--max-requests", type=int)
    p.add_argument("--rpm", type=float)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out-dir", default=str(DATA_DIR))
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    cfg = GenConfig(teacher=a.teacher, examples=a.examples, eval_examples=a.eval_examples,
                    max_usd=a.max_usd, max_requests=a.max_requests, rpm=a.rpm,
                    workers=a.workers, out_dir=a.out_dir, dry_run=a.dry_run)
    if not cfg.dry_run and not (cfg.max_usd or cfg.max_requests):
        print("warning: no --max-usd/--max-requests set; this run has no spending cap", file=sys.stderr)
    m = generate(cfg)
    print(json.dumps({k: m[k] for k in ("teacher", "train", "eval", "requests",
                                         "estimated_spend_usd", "stopped")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
