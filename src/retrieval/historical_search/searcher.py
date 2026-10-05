"""
Historical searcher: semantic search over fight summaries stored in pgvector.

Stores prose summaries of past rounds/fights and retrieves the most relevant
ones given a live event query (e.g. "body shot pressure in late rounds").
"""

import os
from typing import Optional

try:
    from langchain_community.vectorstores import PGVector
    from langchain_community.embeddings import FakeEmbeddings
    try:
        from langchain_openai import OpenAIEmbeddings
        _OPENAI_EMB = True
    except ImportError:
        _OPENAI_EMB = False
    _LANGCHAIN_AVAILABLE = True
    from langchain_core.documents import Document
except ImportError:
    _LANGCHAIN_AVAILABLE = False
    Document = None

_CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:boxio@localhost:5432/boxio",
)
_COLLECTION_NAME = "fight_history"


def _make_embeddings():
    if _OPENAI_EMB and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return FakeEmbeddings(size=1536)


# ── Sample historical fight summaries for seeding ────────────────────────────

SAMPLE_FIGHT_SUMMARIES = [
    {
        "id": "canelo_ggg_i_r4",
        "text": (
            "Round 4: Alvarez began targeting the body relentlessly. Six body shots landed "
            "in 90 seconds, visibly slowing Golovkin's footwork. Golovkin's jab count dropped "
            "from 18/round average to 9 this round. Alvarez's left hook to the liver in the "
            "final 30 seconds was the defining moment, Golovkin winced and stepped back."
        ),
        "tags": ["body_work", "pressure", "late_round_shift"],
    },
    {
        "id": "canelo_ggg_i_r7",
        "text": (
            "Round 7: The momentum pendulum swung. Golovkin's straight rights started finding "
            "the target at a 55% clip. Alvarez's head movement that had been so effective "
            "earlier began slipping, he absorbed 14 punches, his highest of the fight. "
            "Golovkin's punch output surged to 68, suggesting a second-wind response to the body work."
        ),
        "tags": ["momentum_reversal", "counter_punching", "surge"],
    },
    {
        "id": "garcia_campbell_r3",
        "text": (
            "Round 3: Garcia's jab volume spiked to 34, establishing distance against the "
            "taller Campbell. The right hand behind the jab landed 7 times, opening a cut "
            "over Campbell's left eye. Garcia's footwork kept him on the outside, negating "
            "Campbell's natural size and reach advantage."
        ),
        "tags": ["jab_control", "outboxing", "speed_advantage"],
    },
    {
        "id": "canelo_saunders_r8",
        "text": (
            "Round 8: Alvarez landed an uppercut that fractured Saunders' orbital bone, "
            "ending the fight. The shot came off a double jab setup, Saunders reached for "
            "a right hand and left himself exposed inside. Alvarez's punch accuracy peaked "
            "at 72% in this round, a career high for a 3-minute frame."
        ),
        "tags": ["knockout", "combination_finish", "accuracy_peak"],
    },
    {
        "id": "generic_pressure_r10",
        "text": (
            "Championship rounds: pressure fighters tend to take over after round 8. "
            "The cumulative effect of body work slows leg movement; boxers cannot maintain "
            "lateral footwork when the liver area is sore. Counter-punchers who survive "
            "early pressure often find their timing disrupted by accumulated fatigue."
        ),
        "tags": ["late_rounds", "pressure_strategy", "body_work_payoff"],
    },
]


class HistoricalSearcher:
    """
    Stores and retrieves historical round/fight summaries via pgvector.

    Usage:
        searcher = HistoricalSearcher()
        searcher.seed()
        results = searcher.search("pressure fighter dominating body work", k=2)
    """

    def __init__(self):
        self._store: Optional["PGVector"] = None
        self._available = _LANGCHAIN_AVAILABLE
        if not self._available:
            print("WARNING: langchain not available, historical search disabled")

    def _get_store(self) -> Optional["PGVector"]:
        if not self._available:
            return None
        if self._store is None:
            try:
                self._store = PGVector(
                    connection_string=_CONNECTION_STRING,
                    collection_name=_COLLECTION_NAME,
                    embedding_function=_make_embeddings(),
                )
            except Exception as e:
                print(f"WARNING: Historical search store failed ({e})")
                self._available = False
                return None
        return self._store

    def seed(self) -> bool:
        """Insert sample historical summaries. Safe to call multiple times."""
        store = self._get_store()
        if store is None:
            return False

        docs = [
            Document(
                page_content=s["text"],
                metadata={"id": s["id"], "tags": ",".join(s["tags"])},
            )
            for s in SAMPLE_FIGHT_SUMMARIES
        ]
        try:
            store.add_documents(docs)
            return True
        except Exception as e:
            print(f"WARNING: Failed to seed fight history ({e})")
            return False

    def search(self, query: str, k: int = 2) -> list[str]:
        """
        Return top-k historical summaries relevant to `query`.
        Each result is a plain text string.
        """
        store = self._get_store()
        if store is None:
            return []
        try:
            docs = store.similarity_search(query, k=k)
            return [d.page_content for d in docs]
        except Exception as e:
            print(f"WARNING: Historical search failed ({e})")
            return []

    def format_for_prompt(self, query: str, k: int = 2) -> str:
        """Format retrieved summaries as a prompt-injectable block."""
        results = self.search(query, k=k)
        if not results:
            return ""
        parts = ["[Historical Precedents: retrieved for context]"]
        for r in results:
            parts.append(f"• {r}")
        return "\n".join(parts)
