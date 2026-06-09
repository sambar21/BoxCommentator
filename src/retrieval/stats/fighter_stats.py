from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FighterStats:
    name: str
    record_wins: int
    record_losses: int
    record_draws: int
    ko_percentage: float        # 0.0 - 1.0
    reach_inches: float
    stance: str                 # orthodox, southpaw
    style: str                  # pressure, boxer, counter-puncher, brawler, swarmer
    division: str               # heavyweight, welterweight, etc.
    peak_punches_per_round: int
    accuracy_percentage: float  # 0.0 - 1.0
    defense_percentage: float   # 0.0 - 1.0
    body_shot_percentage: float # fraction of offense to body
    notable_wins: list[str] = field(default_factory=list)
    biography: str = ""         # narrative description used for embedding

    @property
    def record_str(self) -> str:
        return f"{self.record_wins}-{self.record_losses}-{self.record_draws}"

    def to_document(self) -> str:
        """Serialize to a prose string suitable for embedding + retrieval."""
        return (
            f"{self.name} ({self.record_str}, {self.division}). "
            f"Style: {self.style}, {self.stance} stance. "
            f"Reach: {self.reach_inches} inches. "
            f"KO rate: {self.ko_percentage:.0%}. "
            f"Accuracy: {self.accuracy_percentage:.0%}, "
            f"Defense: {self.defense_percentage:.0%}. "
            f"Body shot ratio: {self.body_shot_percentage:.0%}. "
            f"Peak output: ~{self.peak_punches_per_round} punches/round. "
            f"Notable wins: {', '.join(self.notable_wins) if self.notable_wins else 'N/A'}. "
            f"{self.biography}"
        )


# ── Sample fighter profiles ───────────────────────────────────────────────────

SAMPLE_FIGHTERS: dict[str, FighterStats] = {
    "Alvarez": FighterStats(
        name="Alvarez",
        record_wins=60, record_losses=2, record_draws=2,
        ko_percentage=0.78,
        reach_inches=70.5,
        stance="orthodox",
        style="pressure",
        division="super-middleweight",
        peak_punches_per_round=52,
        accuracy_percentage=0.58,
        defense_percentage=0.67,
        body_shot_percentage=0.34,
        notable_wins=["Golovkin", "Kovalev", "Saunders"],
        biography=(
            "Aggressive pressure fighter who walks opponents down with relentless body work. "
            "Exceptional chin and counter-punching instincts. Known for a devastating left hook "
            "to the body that has ended numerous bouts. Elite defensive head movement despite "
            "coming forward. Builds pressure in the mid-rounds and is dangerous late when opponents tire."
        ),
    ),
    "Garcia": FighterStats(
        name="Garcia",
        record_wins=40, record_losses=2, record_draws=0,
        ko_percentage=0.85,
        reach_inches=74.0,
        stance="orthodox",
        style="boxer-puncher",
        division="super-middleweight",
        peak_punches_per_round=61,
        accuracy_percentage=0.51,
        defense_percentage=0.61,
        body_shot_percentage=0.22,
        notable_wins=["Vargas", "Easter", "Campbell"],
        biography=(
            "High-output boxer-puncher with exceptional hand speed and a powerful straight right hand. "
            "Prefers to establish the jab and work behind it, creating openings for the overhand right. "
            "Elite footwork when fresh. Susceptible to pressure in later rounds when punch output drops. "
            "Has been stopped twice by body shots — a known vulnerability."
        ),
    ),
    "Johnson": FighterStats(
        name="Johnson",
        record_wins=28, record_losses=3, record_draws=1,
        ko_percentage=0.64,
        reach_inches=76.0,
        stance="southpaw",
        style="counter-puncher",
        division="middleweight",
        peak_punches_per_round=44,
        accuracy_percentage=0.63,
        defense_percentage=0.72,
        body_shot_percentage=0.18,
        notable_wins=["Martinez", "Williams"],
        biography=(
            "Disciplined southpaw counter-puncher with elite accuracy. Patient, rarely leads — "
            "prefers to make opponents miss then land sharp counter left hands. "
            "Uses the southpaw right jab effectively to neutralize orthodox opponents. "
            "Excellent stamina; frequently outworks opponents in the championship rounds."
        ),
    ),
    "Torres": FighterStats(
        name="Torres",
        record_wins=22, record_losses=5, record_draws=0,
        ko_percentage=0.50,
        reach_inches=68.0,
        stance="orthodox",
        style="swarmer",
        division="welterweight",
        peak_punches_per_round=78,
        accuracy_percentage=0.42,
        defense_percentage=0.55,
        body_shot_percentage=0.40,
        notable_wins=["Diaz", "Morales"],
        biography=(
            "High-volume swarmer who overwhelms opponents with output rather than power. "
            "Throws in high-quantity short-range combinations, targeting the body extensively. "
            "Porous defense — gets hit — but relies on an iron chin and will. "
            "Best in close range; struggles when opponents tie up or use movement to stay outside."
        ),
    ),
}
