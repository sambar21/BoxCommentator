"""
Knockdown Tracker - emits a top-priority event when a punch drops a fighter.
"""

from typing import Callable, Dict, Optional
from src.core.action_buffer.buffer import ActionBuffer
from src.core.events import Event


class KnockdownTracker:
    """
    Watches the newest punch in the buffer. Each knockdown punch emits exactly
    one event, even though update() runs on every punch.
    """

    PRIORITY = 10.0

    def __init__(self, buffer: ActionBuffer, names: Optional[Callable[[], Dict[int, str]]] = None):
        self.buffer = buffer
        self._names = names
        self._last_seen = None
        self.count = 0

    def update(self) -> Optional[Event]:
        recent = self.buffer.get_recent(1)
        if not recent:
            return None

        punch = recent[0]
        if punch is self._last_seen:
            return None
        self._last_seen = punch

        if not punch.knockdown:
            return None

        self.count += 1
        attacker = punch.attacker
        victim = 2 if attacker == 1 else 1
        names = self._names() if self._names else {}
        attacker_name = names.get(attacker, f"Fighter {attacker}")
        victim_name = names.get(victim, f"Fighter {victim}")

        return Event(
            type="knockdown",
            priority=self.PRIORITY,
            message=f"KNOCKDOWN! {attacker_name} drops {victim_name} with a {punch.punch_type} to the {punch.target}",
            context={
                "attacker": attacker,
                "victim": victim,
                "punch_type": punch.punch_type,
                "target": punch.target,
                "knockdown_number": self.count,
            },
        )
