"""
Excitement Tracker - detects how exciting the action is.
Triggers idle commentary during lulls.
"""

from typing import Optional
import random
import time
from src.core.action_buffer.buffer import ActionBuffer
from src.core.events import Event

class ExcitementTracker:
    """
    Tracks excitement level based on action density and combo detection.
    """
    
    STATES = ["LULL", "MODERATE", "PEAK"]
    
    def __init__(self, buffer: ActionBuffer):
        self.buffer = buffer
        self.current_state = "MODERATE"
        self.previous_state = None
        
        # Time since last update
        self.last_update_time = time.time()
        
        # Idle timeout (seconds of low activity before triggering idle commentary)
        self.idle_timeout = 4.0
        
        # Thresholds (punches in last 5 seconds)
        self.peak_threshold = 12  # High activity
        self.moderate_threshold = 4  # Normal activity
        
        # Time window to check
        self.time_window = 5.0
        
        # Contexts
        self.contexts = {
            "PEAK": [
                "High-intensity action!",
                "They're going at it!",
                "Explosive exchange!",
                "Fast and furious!"
            ],
            "MODERATE": [
                "Steady action here",
                "Good pace developing"
            ],
            "LULL": [
                "Tactical pause",
                "Both fighters resetting",
                "Moment to breathe"
            ],
            "IDLE": [  # Special idle commentary
                "Let's talk about what we've seen so far",
                "Interesting tactical battle developing",
                "Good technical boxing on display"
            ]
        }
        
        # Priorities
        self.priorities = {
            "PEAK": 9.5,      # Excitement is important!
            "MODERATE": 3.0,
            "LULL": 3.0,
            "IDLE": 1.0       # Low priority (filler)
        }
    
    def update(self) -> Optional[Event]:
        """Check excitement level"""
        current_time = time.time()
        new_state = self._calculate_state()
        
        # Check for idle timeout (if in LULL for too long)
        if new_state == "LULL":
            time_since_update = current_time - self.last_update_time
            
            if time_since_update >= self.idle_timeout:
                # Trigger idle commentary
                self.last_update_time = current_time
                return self._create_idle_event()
        
        # Normal state change
        if new_state != self.current_state:
            event = self._create_event(new_state)
            self.previous_state = self.current_state
            self.current_state = new_state
            self.last_update_time = current_time
            return event
        
        return None
    
    def _calculate_state(self) -> str:
        """Determine excitement level"""
        recent_punches = self.buffer.get_recent_seconds(self.time_window)
        punch_count = len(recent_punches)
        
        if punch_count >= self.peak_threshold:
            return "PEAK"
        elif punch_count >= self.moderate_threshold:
            return "MODERATE"
        else:
            return "LULL"
    
    def _create_event(self, new_state: str) -> Event:
        """Create excitement change event"""
        event_type = "excitement_peak" if new_state == "PEAK" else "excitement_change"
        
        return Event(
            type=event_type,
            priority=self.priorities[new_state],
            message=random.choice(self.contexts[new_state]),
            context={
                "from_state": self.current_state,
                "to_state": new_state,
                "recent_punch_count": len(self.buffer.get_recent_seconds(self.time_window))
            }
        )
    
    def _create_idle_event(self) -> Event:
        """Create idle commentary event (filler during lulls)"""
        return Event(
            type="idle_timeout",
            priority=self.priorities["IDLE"],
            message=random.choice(self.contexts["IDLE"]),
            context={
                "reason": "idle_timeout",
                "time_since_action": time.time() - self.last_update_time
            }
        )