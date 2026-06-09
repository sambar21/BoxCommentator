"""
FastAPI routes — HTTP wrapper around CommentaryOrchestrator.
Called by the Go gateway via HTTP.
"""

import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.core.action_buffer.buffer import Punch
from src.core.orchestrator import CommentaryOrchestrator

router = APIRouter()

# Singleton orchestrator shared across requests
_orchestrator: Optional[CommentaryOrchestrator] = None


def get_orchestrator() -> CommentaryOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = CommentaryOrchestrator()
    return _orchestrator


# ── Request / Response models ─────────────────────────────────────────────────

class PunchRequest(BaseModel):
    attacker: int           # 1 or 2
    punch_type: str         # jab, cross, hook, uppercut
    target: str             # head, body
    outcome: str            # landed, missed, blocked
    damage: int = 0
    fighter1_name: str = "Fighter 1"
    fighter2_name: str = "Fighter 2"
    round_num: int = 1

    def to_domain(self) -> Punch:
        return Punch(
            attacker=self.attacker,
            punch_type=self.punch_type,
            target=self.target,
            outcome=self.outcome,
            timestamp=time.time(),
            damage=self.damage,
        )


class CommentaryResponse(BaseModel):
    commentary: Optional[str]
    latency_ms: float
    p95_latency_ms: float
    provider: str


class HealthResponse(BaseModel):
    status: str
    rag_available: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/internal/punch", response_model=CommentaryResponse)
async def process_punch(req: PunchRequest) -> CommentaryResponse:
    t0 = time.perf_counter()
    orch = get_orchestrator()

    # Lazy fight setup: set names and start round on first punch if needed
    names = orch.context_builder.fighter_names
    if names[1] != req.fighter1_name or names[2] != req.fighter2_name:
        orch.set_fighter_names(req.fighter1_name, req.fighter2_name)
        orch.start_round(req.round_num)

    try:
        commentary = orch.process_punch(req.to_domain())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000

    return CommentaryResponse(
        commentary=commentary,
        latency_ms=round(latency_ms, 2),
        p95_latency_ms=round(orch.get_p95_latency(), 2),
        provider=orch.llm.get_provider_name(),
    )


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    orch = get_orchestrator()
    rag_ok = orch._fight_memory is not None
    return HealthResponse(status="ok", rag_available=rag_ok)
