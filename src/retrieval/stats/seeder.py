"""
Seed pgvector with fighter profiles and historical summaries.
Run once before a fight: python -m src.retrieval.stats.seeder Alvarez Garcia
"""

import sys
from dotenv import load_dotenv
load_dotenv()

from src.retrieval.fight_memory.store import FightMemoryStore
from src.retrieval.historical_search.searcher import HistoricalSearcher


def seed(fighter_names: list[str]):
    print(f"Seeding fighter profiles: {fighter_names}")
    store = FightMemoryStore()
    ok = store.seed_fighters(fighter_names)
    print(f"  Fighter profiles: {'OK' if ok else 'FAILED (check DATABASE_URL and langchain install)'}")

    print("Seeding historical fight summaries...")
    history = HistoricalSearcher()
    ok = history.seed()
    print(f"  Fight history: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    names = sys.argv[1:] if len(sys.argv) > 1 else ["Alvarez", "Garcia"]
    seed(names)
