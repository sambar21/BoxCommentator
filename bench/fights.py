"""
Seeded fight simulator with ground-truth knockdowns.

make_fight(seed, ...) is fully deterministic: same seed -> same fight, so every
backend commentates identical punches and quality.py can check against truth.

replay(fight, orchestrator) feeds the fight through an orchestrator on a
virtual clock, so a 3-minute round replays in milliseconds of fight-logic time
(LLM calls still take real wall-clock time, which is what we measure).

CLI: python -m bench.fights --seed 1
"""

import argparse
import random
from dataclasses import dataclass, field
from typing import List, Optional

from src.core import clock
from src.core.action_buffer.buffer import Punch
from src.retrieval.stats.fighter_stats import SAMPLE_FIGHTERS

ROUND_SECONDS = 180.0
PUNCH_TYPES = ["jab", "cross", "hook", "uppercut"]
POWER_PUNCHES = ["cross", "hook", "uppercut"]


@dataclass
class PunchSpec:
    t: float            # seconds since fight start
    round_num: int
    attacker: int
    punch_type: str
    target: str
    outcome: str
    damage: int
    knockdown: bool = False


@dataclass
class Fight:
    fight_id: str
    seed: int
    fighter1: str
    fighter2: str
    punches: List[PunchSpec]
    knockdown_indices: List[int] = field(default_factory=list)  # indices into punches

    @property
    def n_knockdowns(self) -> int:
        return len(self.knockdown_indices)

    @property
    def duration(self) -> float:
        return self.punches[-1].t if self.punches else 0.0


def _weighted(rng: random.Random, options, weights):
    return rng.choices(options, weights=weights, k=1)[0]


def make_fight(
    seed: int,
    n_punches: int = 120,
    n_knockdowns: int = 1,
    fighter1: Optional[str] = None,
    fighter2: Optional[str] = None,
) -> Fight:
    rng = random.Random(seed)
    names = list(SAMPLE_FIGHTERS)
    f1 = fighter1 or rng.choice(names)
    f2 = fighter2 or rng.choice([n for n in names if n != f1])
    stats = {1: SAMPLE_FIGHTERS[f1], 2: SAMPLE_FIGHTERS[f2]}

    # Pick knockdown slots away from the start/end and from each other.
    n_knockdowns = min(n_knockdowns, max(0, n_punches // 20))
    lo, hi = max(5, n_punches // 10), n_punches - 3
    slots: List[int] = []
    tries = 0
    while len(slots) < n_knockdowns and tries < 1000:
        tries += 1
        idx = rng.randint(lo, hi)
        if all(abs(idx - s) >= 12 for s in slots):
            slots.append(idx)
    slots.sort()

    punches: List[PunchSpec] = []
    t = 0.0
    burst_left = 0
    lull_left = 0
    momentum_owner = rng.choice([1, 2])

    for i in range(n_punches):
        # Alternate bursts (fast exchanges) and lulls (tactical pauses).
        if burst_left == 0 and lull_left == 0:
            if rng.random() < 0.55:
                burst_left = rng.randint(6, 16)
            else:
                lull_left = rng.randint(2, 5)
            if rng.random() < 0.3:
                momentum_owner = 3 - momentum_owner

        if burst_left > 0:
            gap = rng.uniform(0.25, 0.9)
            burst_left -= 1
        else:
            gap = rng.uniform(2.0, 4.5)
            lull_left -= 1
        t += gap

        is_knockdown = i in slots
        # Momentum owner throws ~65% of punches; the knockdown is scored by them.
        attacker = momentum_owner if rng.random() < 0.65 else 3 - momentum_owner
        a = stats[attacker]

        if is_knockdown:
            punch_type = rng.choice(POWER_PUNCHES)
            target = "head" if punch_type != "uppercut" or rng.random() < 0.7 else "body"
            outcome = "landed"
            damage = rng.randint(40, 60)
            t += rng.uniform(4.0, 8.0)  # the count: dead air after the knockdown
        else:
            punch_type = _weighted(rng, PUNCH_TYPES, [10, 5, 5, 2])
            target = "body" if rng.random() < a.body_shot_percentage else "head"
            outcome = "landed" if rng.random() < a.accuracy_percentage else (
                "blocked" if rng.random() < 0.5 else "missed")
            damage = rng.randint(5, 25) if outcome == "landed" else 0

        punches.append(PunchSpec(
            t=round(t, 3),
            round_num=int(t // ROUND_SECONDS) + 1,
            attacker=attacker,
            punch_type=punch_type,
            target=target,
            outcome=outcome,
            damage=damage,
            knockdown=is_knockdown,
        ))

    return Fight(
        fight_id=f"fight-{seed:04d}",
        seed=seed,
        fighter1=f1,
        fighter2=f2,
        punches=punches,
        knockdown_indices=slots,
    )


def make_suite(n_fights: int = 30, base_seed: int = 1000) -> List[Fight]:
    """
    The benchmark suite: a mix of short fights, standard fights and long fights
    (long ones exercise the context-capping experiment). Every fight has 1-3
    knockdowns so knockdown recall has a meaningful denominator.
    """
    fights = []
    for i in range(n_fights):
        kind = i % 3
        if kind == 0:
            n_punches, n_kd = 60, 1
        elif kind == 1:
            n_punches, n_kd = 150, 2
        else:
            n_punches, n_kd = 400, 3
        fights.append(make_fight(base_seed + i, n_punches=n_punches, n_knockdowns=n_kd))
    return fights


def replay(fight: Fight, orchestrator, virtual: Optional[clock.VirtualClock] = None,
           on_punch=None):
    """
    Feed a fight through `orchestrator` on a virtual clock.

    Returns the list of (punch_index, commentary) pairs the orchestrator produced.
    Restores the real clock afterwards. Build the orchestrator *after* the clock
    is installed if you create it yourself; this function installs the clock
    before touching it, so pass a freshly constructed orchestrator.
    """
    virtual = virtual or clock.VirtualClock()
    clock.set_source(virtual.now)
    random.seed(fight.seed)   # tracker message choice is random; keep sequential runs identical
    outputs = []
    try:
        start = virtual.now()
        orchestrator.set_fighter_names(fight.fighter1, fight.fighter2)
        current_round = None
        for i, spec in enumerate(fight.punches):
            virtual.set(start + spec.t)
            if spec.round_num != current_round:
                current_round = spec.round_num
                orchestrator.start_round(current_round)
            punch = Punch(
                attacker=spec.attacker,
                punch_type=spec.punch_type,
                target=spec.target,
                outcome=spec.outcome,
                timestamp=virtual.now(),
                damage=spec.damage,
                knockdown=spec.knockdown,
            )
            text = orchestrator.process_punch(punch)
            if text:
                outputs.append((i, text))
            if on_punch:
                on_punch(i, spec, text)
    finally:
        clock.reset()
    return outputs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--punches", type=int, default=120)
    ap.add_argument("--knockdowns", type=int, default=2)
    args = ap.parse_args()

    f = make_fight(args.seed, args.punches, args.knockdowns)
    print(f"{f.fight_id}: {f.fighter1} vs {f.fighter2}, {len(f.punches)} punches, "
          f"{f.duration:.0f}s, knockdowns at {f.knockdown_indices}")
    for i in f.knockdown_indices:
        p = f.punches[i]
        print(f"  #{i} t={p.t:.1f}s P{p.attacker} {p.punch_type} to {p.target} (dmg {p.damage})")
