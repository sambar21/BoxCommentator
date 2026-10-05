"""
Commentate a simulated fight out loud and log time-to-first-audio.

  python -m voice.speak_fight --dry-run                    # stub LLM + fake voice, no models
  python -m voice.speak_fight --backend groq-qwen27b --fights 1 --play
  python -m voice.speak_fight --backend vllm-3b --fights 2 --max-lines 40

Each commentary line from the orchestrator is rendered with Kokoro; the CSV has one row per line:
LLM time, TTS first-audio time, end-to-end first-audio time and real-time factor.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Optional

from bench.bench import Budget, MeteredLLM, STUB_BACKEND, percentile
from bench.fights import make_suite, replay
from bench.stub_llm import StubLLM
from src.core.orchestrator import CommentaryOrchestrator
from voice.kokoro_tts import KokoroEngine, Player, SpeechTiming, speak

OUT_DIR = Path(__file__).parent / "results"


class FakeEngine:
    """Stands in for Kokoro in dry runs and tests: instant, silent audio."""
    sample_rate = 24000

    def __init__(self, delay_s: float = 0.0):
        self.delay_s = delay_s

    def synthesize(self, text: str):
        import time
        import numpy as np
        for _ in text.split(". "):
            time.sleep(self.delay_s)
            yield np.zeros(self.sample_rate // 2, dtype="float32")


def run(llm, engine, fights: int, max_lines: Optional[int], player: Optional[Player]) -> List[dict]:
    rows = []
    for fight in make_suite(30)[:fights]:
        orch = CommentaryOrchestrator(llm_client=llm, use_rag=False)
        replay(fight, orch)
        for rec in orch.generation_log:
            if not rec.text or (max_lines and len(rows) >= max_lines):
                continue
            t: SpeechTiming = speak(rec.text, engine, llm_ms=rec.total_ms,
                                    on_audio=player.play if player else None)
            rows.append({"fight_id": fight.fight_id, "track": rec.track, "event_type": rec.event_type,
                         **{k: v for k, v in t.to_dict().items()}})
    return rows


def summarize(rows: List[dict]) -> dict:
    tta = [r["tts_first_audio_ms"] for r in rows if r["tts_first_audio_ms"] is not None]
    e2e = [r["e2e_first_audio_ms"] for r in rows if r["e2e_first_audio_ms"] is not None]
    rtf = [r["real_time_factor"] for r in rows if r["real_time_factor"] is not None]
    return {
        "lines": len(rows),
        "tts_first_audio_p50_ms": percentile(tta, 50), "tts_first_audio_p95_ms": percentile(tta, 95),
        "e2e_first_audio_p50_ms": percentile(e2e, 50), "e2e_first_audio_p95_ms": percentile(e2e, 95),
        "real_time_factor_p50": percentile(rtf, 50),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", help="backend name from config/backends.yaml")
    p.add_argument("--fights", type=int, default=1)
    p.add_argument("--max-lines", type=int, default=None)
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--play", action="store_true", help="play the audio (needs pygame)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=str(OUT_DIR))
    a = p.parse_args(argv)

    if a.dry_run:
        llm, engine = MeteredLLM(StubLLM(), STUB_BACKEND, Budget()), FakeEngine()
    else:
        from src.generation.llm_interface.llm_factory import LLMFactory
        if not a.backend:
            raise SystemExit("pass --backend NAME (see: python -m bench.bench --list) or --dry-run")
        llm = LLMFactory.create_client(a.backend)
        engine = KokoroEngine(voice=a.voice)
        engine.warmup()
    player = Player() if a.play else None

    rows = run(llm, engine, a.fights, a.max_lines, player)
    if player:
        player.wait()
    if not rows:
        print("no commentary lines were produced", file=sys.stderr)
        return 1

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("dry_run.csv" if a.dry_run else f"{a.backend}_voice.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for k, v in summarize(rows).items():
        print(f"  {k:26} {v if not isinstance(v, float) else round(v, 1)}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
