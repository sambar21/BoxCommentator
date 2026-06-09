"""
Sliding window analyzer — computes live per-round stats from the ActionBuffer.

These stats are injected directly into the LLM prompt (no embedding needed —
they're computed, not retrieved). Gives the LLM real numbers to work with.
"""

from src.core.action_buffer.buffer import ActionBuffer, Punch


class SlidingWindowAnalyzer:
    """
    Computes real-time fight statistics from the ActionBuffer.

    Returns a structured stats dict ready for prompt injection.
    """

    def __init__(self, buffer: ActionBuffer):
        self.buffer = buffer

    def compute(self, window_seconds: float = 10.0) -> dict:
        """
        Compute stats over the last `window_seconds` of the fight.

        Returns dict with keys usable directly in the LLM prompt.
        """
        recent = self.buffer.get_recent_seconds(window_seconds)
        all_punches = self.buffer.get_all()

        return {
            "window_seconds": window_seconds,
            "recent": self._player_stats(recent),
            "overall": self._player_stats(all_punches),
            "pace_per_minute": round(len(recent) / (window_seconds / 60), 1),
        }

    def _player_stats(self, punches: list[Punch]) -> dict:
        if not punches:
            return {
                "p1": self._empty_stats(),
                "p2": self._empty_stats(),
                "total": 0,
            }

        p1 = [p for p in punches if p.attacker == 1]
        p2 = [p for p in punches if p.attacker == 2]

        return {
            "p1": self._calc(p1),
            "p2": self._calc(p2),
            "total": len(punches),
        }

    def _calc(self, punches: list[Punch]) -> dict:
        if not punches:
            return self._empty_stats()

        landed = [p for p in punches if p.outcome == "landed"]
        head = [p for p in landed if p.target == "head"]
        body = [p for p in landed if p.target == "body"]
        total_damage = sum(p.damage for p in landed)

        accuracy = len(landed) / len(punches) if punches else 0.0

        punch_type_counts: dict[str, int] = {}
        for p in punches:
            punch_type_counts[p.punch_type] = punch_type_counts.get(p.punch_type, 0) + 1
        top_punch = max(punch_type_counts, key=punch_type_counts.get) if punch_type_counts else "N/A"

        return {
            "thrown": len(punches),
            "landed": len(landed),
            "accuracy": round(accuracy * 100, 1),
            "head_landed": len(head),
            "body_landed": len(body),
            "body_ratio": round(len(body) / len(landed) * 100, 1) if landed else 0.0,
            "total_damage": total_damage,
            "top_punch_type": top_punch,
        }

    def _empty_stats(self) -> dict:
        return {
            "thrown": 0, "landed": 0, "accuracy": 0.0,
            "head_landed": 0, "body_landed": 0, "body_ratio": 0.0,
            "total_damage": 0, "top_punch_type": "N/A",
        }

    def format_for_prompt(
        self,
        p1_name: str = "Fighter 1",
        p2_name: str = "Fighter 2",
        window_seconds: float = 10.0,
    ) -> str:
        """Return a compact stats block for injection into the LLM prompt."""
        s = self.compute(window_seconds)
        r = s["recent"]
        return (
            f"[Live Stats — last {window_seconds:.0f}s]\n"
            f"{p1_name}: {r['p1']['landed']}/{r['p1']['thrown']} landed "
            f"({r['p1']['accuracy']}% acc), "
            f"body {r['p1']['body_ratio']}%, top punch: {r['p1']['top_punch_type']}\n"
            f"{p2_name}: {r['p2']['landed']}/{r['p2']['thrown']} landed "
            f"({r['p2']['accuracy']}% acc), "
            f"body {r['p2']['body_ratio']}%, top punch: {r['p2']['top_punch_type']}\n"
            f"Pace: {s['pace_per_minute']} punches/min"
        )
