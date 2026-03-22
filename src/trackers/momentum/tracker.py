"""
Momentum Tracker - detects who's improving vs declining.
Compares recent performance to previous performance.
"""

from typing import Optional
import random
from src.core.action_buffer.buffer import ActionBuffer
from src.core.events import Event

class MomentumTracker:
    """
    Tracks momentum shifts - who's getting better vs worse.
    """
    
    STATES = ["P2_RISING", "STABLE", "P1_RISING"]
    
    def __init__(self, buffer: ActionBuffer):
        self.buffer = buffer
        self.current_state = "STABLE"
        self.previous_state = None
        
        # How many punches to compare
        self.window_size = 10
        
        # Threshold for momentum shift (punch difference)
        self.shift_threshold = 3
        
        # Contexts
        self.contexts = {
            "P2_RISING": [
                "Player 2 is building momentum!",
                "Player 2 mounting a comeback",
                "The tide is turning in Player 2's favor",
                "Player 2 surging now"
            ],
            "P1_RISING": [
                "Player 1 building momentum!",
                "Player 1 mounting a comeback",
                "The tide is turning in Player 1's favor",
                "Player 1 surging now"
            ],
            "STABLE": [
                "Momentum has stabilized",
                "Back and forth action",
                "Neither fighter gaining clear advantage"
            ]
        }
        
        # Priorities
        self.priorities = {
            "P2_RISING": 7.5,
            "P1_RISING": 7.5,
            "STABLE": 4.0
        }
    
    def update(self) -> Optional[Event]:
        """Check for momentum shifts"""
        new_state = self._calculate_state()
        
        if new_state != self.current_state:
            event = self._create_event(new_state)
            self.previous_state = self.current_state
            self.current_state = new_state
            return event
        
        return None
    
    def _calculate_state(self) -> str:
        """Determine current momentum"""
        all_punches = self.buffer.get_all()
        
        # Need at least 2x window size to compare
        if len(all_punches) < self.window_size * 2:
            return "STABLE"
        
        # Split into recent and previous windows
        recent = all_punches[-self.window_size:]
        previous = all_punches[-self.window_size*2:-self.window_size]
        
        # Count landed punches in each window
        p1_recent = sum(1 for p in recent if p.attacker == 1 and p.outcome == 'landed')
        p2_recent = sum(1 for p in recent if p.attacker == 2 and p.outcome == 'landed')
        
        p1_previous = sum(1 for p in previous if p.attacker == 1 and p.outcome == 'landed')
        p2_previous = sum(1 for p in previous if p.attacker == 2 and p.outcome == 'landed')
        
        # Calculate momentum change
        p1_change = p1_recent - p1_previous
        p2_change = p2_recent - p2_previous
        
        momentum_diff = p1_change - p2_change
        
        # Determine state
        if momentum_diff >= self.shift_threshold:
            return "P1_RISING"
        elif momentum_diff <= -self.shift_threshold:
            return "P2_RISING"
        else:
            return "STABLE"
    
    def _create_event(self, new_state: str) -> Event:
        """Create momentum shift event"""
        return Event(
            type="momentum_reversal" if new_state != "STABLE" else "momentum_stable",
            priority=self.priorities[new_state],
            message=random.choice(self.contexts[new_state]),
            context={
                "from_state": self.current_state,
                "to_state": new_state
            }
        )