"""
Pace Tracker - Monitors fight tempo and speed.
Uses hysteresis to prevent state oscillation.
"""

from typing import Optional, List
from src.core.action_buffer.buffer import ActionBuffer
from src.core.events import Event
import random


class PaceTracker:
    STATES = ["SLOW", "MODERATE", "FAST"]
    
    def __init__(self, buffer: ActionBuffer):
        self.buffer = buffer
        self.current_state = "MODERATE"  # Start assumption
        self.previous_state = None
        
        # Thresholds with hysteresis
        self.fast_threshold = 18      # Need 18+ to enter FAST
        self.fast_exit = 14           # Need <14 to exit FAST
        
        self.moderate_threshold = 8   # Need 8+ to enter MODERATE
        self.moderate_exit = 5        # Need <5 to exit MODERATE
        
        # Time window to check
        self.time_window = 10.0  # seconds
        
        # Contexts for each state
        self.contexts = {
            "SLOW": [
                "This is a tactical, measured pace",
                "Both fighters being patient here",
                "Slow, technical boxing"
            ],
            "MODERATE": [
                "Good steady pace to this fight",
                "Nice rhythm developing",
                "Maintaining a consistent pace"
            ],
            "FAST": [
                "Fast-paced action here!",
                "High tempo battle",
                "They're letting their hands go!"
            ]
        }
        
        # Priorities for pace changes
        self.priorities = {
            "SLOW": 2.0,      # Low priority (less exciting)
            "MODERATE": 3.0,
            "FAST": 6.0       # Higher priority (exciting!)
        }
    
    def update(self) -> Event | None:
        """Check pace and emit event if changed"""
        new_state = self._calculate_state()
        
        if new_state != self.current_state:
            event = self._create_event(new_state)
            self.previous_state = self.current_state
            self.current_state = new_state
            return event
        
        return None
    
    def _calculate_state(self) -> str:
        """
        Determine current pace based on recent punches.
        Uses hysteresis (different thresholds for entry/exit) to prevent oscillation.
        """
        # Get punches from last N seconds
        recent_punches = self.buffer.get_recent_seconds(self.time_window)
        punch_count = len(recent_punches)
        
        # Use hysteresis (different thresholds for entry/exit)
        if self.current_state == "FAST":
            # Already fast - need to drop below exit threshold to leave
            return "FAST" if punch_count >= self.fast_exit else self._determine_lower_state(punch_count)
        
        elif self.current_state == "MODERATE":
            # In moderate - check both directions
            if punch_count >= self.fast_threshold:
                return "FAST"
            elif punch_count < self.moderate_exit:
                return "SLOW"
            else:
                return "MODERATE"
        
        else:  # SLOW
            # In slow - need to exceed threshold to leave
            if punch_count >= self.fast_threshold:
                return "FAST"
            elif punch_count >= self.moderate_threshold:
                return "MODERATE"
            else:
                return "SLOW"
    
    def _determine_lower_state(self, punch_count: int) -> str:
        """Helper: When exiting FAST, where do we go?"""
        if punch_count >= self.moderate_threshold:
            return "MODERATE"
        else:
            return "SLOW"
    
    def _create_event(self, new_state: str) -> Event:
        """Create pace change event"""
        return Event(
            type="pace_change",
            priority=self.priorities[new_state],
            message=random.choice(self.contexts[new_state]),
            context={
                "from_state": self.current_state,
                "to_state": new_state,
                "punch_count": len(self.buffer.get_recent_seconds(self.time_window)),
                "time_window": self.time_window
            }
        )