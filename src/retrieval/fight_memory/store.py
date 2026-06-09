"""
PGVector store — stores fighter profiles and fight history as embeddings.
Uses LangChain's PGVector integration for similarity search.

Fallback: if pgvector / psycopg2 unavailable, returns empty docs so the
rest of the pipeline degrades gracefully (commentary still works, just
without retrieved stats).
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
except ImportError:
    _LANGCHAIN_AVAILABLE = False

try:
    from langchain_core.documents import Document
except ImportError:
    Document = None

from src.retrieval.stats.fighter_stats import FighterStats, SAMPLE_FIGHTERS


_CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:boxio@localhost:5432/boxio",
)
_COLLECTION_NAME = "boxer_profiles"


def _make_embeddings():
    """Pick embeddings: OpenAI if key available, else deterministic fakes."""
    if _OPENAI_EMB and os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    # FakeEmbeddings produces consistent-length vectors — good enough for
    # local dev / tests where exact semantic quality doesn't matter.
    return FakeEmbeddings(size=1536)


class FightMemoryStore:
    """
    Thin wrapper around LangChain PGVector.

    Usage:
        store = FightMemoryStore()
        store.seed_fighters(["Alvarez", "Garcia"])
        docs = store.retrieve("body shot pressure fighter", k=2)
    """

    def __init__(self):
        self._store: Optional["PGVector"] = None
        self._available = _LANGCHAIN_AVAILABLE
        if not self._available:
            print(
                "WARNING: langchain-community not installed — "
                "RAG retrieval disabled. pip install langchain-community langchain-openai"
            )

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
                print(f"WARNING: PGVector connection failed ({e}) — RAG disabled")
                self._available = False
                return None
        return self._store

    def seed_fighters(self, fighter_names: list[str]) -> bool:
        """
        Embed and upsert fighter profiles for the given names.
        Looks up SAMPLE_FIGHTERS; unknown names are silently skipped.
        Returns True if at least one profile was stored.
        """
        store = self._get_store()
        if store is None:
            return False

        docs = []
        for name in fighter_names:
            profile = SAMPLE_FIGHTERS.get(name)
            if profile is None:
                continue
            docs.append(
                Document(
                    page_content=profile.to_document(),
                    metadata={
                        "fighter": name,
                        "style": profile.style,
                        "stance": profile.stance,
                        "ko_pct": profile.ko_percentage,
                    },
                )
            )

        if not docs:
            return False

        try:
            store.add_documents(docs)
            return True
        except Exception as e:
            print(f"WARNING: Failed to seed fighters ({e})")
            return False

    def retrieve(self, query: str, k: int = 3) -> list:
        """
        Retrieve top-k fighter documents most semantically similar to query.
        Returns empty list on failure — callers must handle gracefully.
        """
        store = self._get_store()
        if store is None:
            return []
        try:
            return store.similarity_search(query, k=k)
        except Exception as e:
            print(f"WARNING: RAG retrieval failed ({e})")
            return []

    def format_for_prompt(self, query: str, k: int = 3) -> str:
        """
        Retrieve and format retrieved fighter stats as a prompt-injectable string.
        Returns empty string if nothing retrieved.
        """
        docs = self.retrieve(query, k=k)
        if not docs:
            return ""
        parts = [f"[Fighter Stats — retrieved for context]\n"]
        for doc in docs:
            parts.append(f"• {doc.page_content}")
        return "\n".join(parts)
