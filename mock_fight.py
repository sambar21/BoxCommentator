"""
Mock Fight Demo - Showcases the full AI commentary pipeline.

Simulates a boxing round with randomized punches and runs them through
the CommentaryOrchestrator, which coordinates all trackers, the priority
queue, and dual-track LLM generation.
"""

import time
import random
from dotenv import load_dotenv
load_dotenv()

from src.core.action_buffer.buffer import Punch
from src.core.orchestrator import CommentaryOrchestrator


def run_mock_fight():
    orchestrator = CommentaryOrchestrator()
    orchestrator.set_fighter_names("Alvarez", "Garcia")
    orchestrator.start_round(1)

    punch_types = ["jab"] * 10 + ["cross"] * 5 + ["hook"] * 5 + ["uppercut"] * 3
    targets = ["head", "body"]
    outcomes = ["landed"] * 6 + ["missed"] * 2 + ["blocked"] * 2

    print("=" * 60)
    print(" BOX.IO AI COMMENTARY - MOCK FIGHT")
    print("=" * 60 + "\n")

    for sec in range(1, 61):
        if random.random() > 0.4:
            punch = Punch(
                attacker=random.choice([1, 2]),
                punch_type=random.choice(punch_types),
                target=random.choice(targets),
                outcome=random.choice(outcomes),
                timestamp=time.time(),
                damage=random.randint(5, 25),
            )

            label = f"P{punch.attacker} {punch.punch_type} -> {punch.target} ({punch.outcome})"
            print(f"  T={sec:02d}s | {label}")

            commentary = orchestrator.process_punch(punch)
            if commentary:
                print(f"\n  >>> {commentary}\n")
        else:
            print(f"  T={sec:02d}s | ---")

        time.sleep(0.3)

    stats = orchestrator.get_stats()
    print("\n" + "=" * 60)
    print(" ROUND STATS")
    print("=" * 60)
    print(f"  Punches processed : {stats['total_punches']}")
    print(f"  Events emitted    : {stats['total_events_emitted']}")
    print(f"  Commentary lines  : {stats['commentary_generated']}")
    print(f"    Track A         : {stats['track_a_generated']}")
    print(f"    Track B         : {stats['track_b_generated']}")
    print(f"  Dominance         : {stats['current_dominance']}")
    print(f"  Pace              : {stats['current_pace']}")
    print(f"  Momentum          : {stats['current_momentum']}")
    print("=" * 60)


if __name__ == "__main__":
    run_mock_fight()
