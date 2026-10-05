"""
FastAPI routes: HTTP wrapper around CommentaryOrchestrator.
Called by the Go gateway via HTTP.

One orchestrator per fight_id, so concurrent fights never share tracker, queue
or buffer state. Generation is blocking LLM I/O, so it runs in a worker thread
(guarded by a per-fight lock; punches within one fight are processed in order).
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from src.core import clock
from src.core.action_buffer.buffer import Punch
from src.core.orchestrator import CommentaryOrchestrator

router = APIRouter()

MAX_FIGHTS = 256  # evict least-recently-used fights beyond this


@dataclass
class FightSession:
    orchestrator: CommentaryOrchestrator
    lock: threading.Lock = field(default_factory=threading.Lock)
    names: tuple = ()
    round_num: Optional[int] = None
    last_used: float = field(default_factory=time.time)


_sessions: Dict[str, FightSession] = {}
_sessions_lock = threading.Lock()


def get_session(fight_id: str) -> FightSession:
    with _sessions_lock:
        session = _sessions.get(fight_id)
        if session is None:
            if len(_sessions) >= MAX_FIGHTS:
                oldest = min(_sessions, key=lambda k: _sessions[k].last_used)
                del _sessions[oldest]
            session = FightSession(orchestrator=CommentaryOrchestrator())
            _sessions[fight_id] = session
        session.last_used = time.time()
        return session


def reset_sessions():
    """Drop all fight state (tests)."""
    with _sessions_lock:
        _sessions.clear()


# ── Request / Response models ─────────────────────────────────────────────────

class PunchRequest(BaseModel):
    attacker: int           # 1 or 2
    punch_type: str         # jab, cross, hook, uppercut
    target: str             # head, body
    outcome: str            # landed, missed, blocked
    damage: int = 0
    knockdown: bool = False
    fight_id: str = "default"
    fighter1_name: str = "Fighter 1"
    fighter2_name: str = "Fighter 2"
    round_num: int = 1

    def to_domain(self) -> Punch:
        return Punch(
            attacker=self.attacker,
            punch_type=self.punch_type,
            target=self.target,
            outcome=self.outcome,
            timestamp=clock.now(),
            damage=self.damage,
            knockdown=self.knockdown,
        )


class CommentaryResponse(BaseModel):
    commentary: Optional[str]
    latency_ms: float
    p95_latency_ms: float
    provider: str
    fight_id: str = "default"


class HealthResponse(BaseModel):
    status: str
    rag_available: bool
    active_fights: int = 0


# ── Endpoints ─────────────────────────────────────────────────────────────────

def _process(session: FightSession, req: PunchRequest) -> Optional[str]:
    with session.lock:
        orch = session.orchestrator
        names = (req.fighter1_name, req.fighter2_name)
        if session.names != names:
            orch.set_fighter_names(*names)
            session.names = names
            session.round_num = None
        if session.round_num != req.round_num:
            orch.start_round(req.round_num)
            session.round_num = req.round_num
        return orch.process_punch(req.to_domain())


@router.post("/internal/punch", response_model=CommentaryResponse)
async def process_punch(req: PunchRequest) -> CommentaryResponse:
    t0 = time.perf_counter()
    session = get_session(req.fight_id)

    try:
        commentary = await run_in_threadpool(_process, session, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000

    return CommentaryResponse(
        commentary=commentary,
        latency_ms=round(latency_ms, 2),
        p95_latency_ms=round(session.orchestrator.get_p95_latency(), 2),
        provider=session.orchestrator.llm.get_provider_name(),
        fight_id=req.fight_id,
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    with _sessions_lock:
        sessions = list(_sessions.values())
    rag_ok = any(s.orchestrator._fight_memory is not None for s in sessions) if sessions else True
    return HealthResponse(status="ok", rag_available=rag_ok, active_fights=len(sessions))
