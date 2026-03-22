"""
Target Zone Tracker - tracks where punches are landing (head vs body).
"""

from typing import Optional
import random
from src.core.action_buffer.buffer import ActionBuffer
from src.core.events import Event

class TargetZoneTracker:
    """
    Tracks attack targeting - head hunting vs body work.
    """
    
    STATES = ["BODY_FOCUS", "MIXED", "HEAD_FOCUS"]
    
    def __init__(self, buffer: ActionBuffer):
        self.buffer = buffer
        self.current_state = "MIXED"
        self.previous_state = None
        
        # Threshold for focused attack (% of punches to one zone)
        self.focus_threshold = 0.70  # 70% to one zone = focused
        
        # Minimum punches to make judgment
        self.min_punches = 5
        
        # Contexts
        self.contexts = {
            "BODY_FOCUS": [
                "Working the body systematically",
                "Attacking the body",
                "Good body work here",
                "Focusing on body shots"
            ],
            "HEAD_FOCUS": [
                "Head hunting now",
                "Looking for the head shot",
                "All upstairs attacks",
                "Targeting the head"
            ],
            "MIXED": [
                "Mixing up the attack",
                "Varying the targets",
                "Good combination of head and body"
            ]
        }
        
        # Priorities
        self.priorities = {
            "BODY_FOCUS": 6.0,
            "HEAD_FOCUS": 6.0,
            "MIXED": 4.0
        }
    
    def update(self) -> Optional[Event]:
        """Check target zone patterns"""
        new_state = self._calculate_state()
        
        if new_state != self.current_state:
            event = self._create_event(new_state)
            self.previous_state = self.current_state
            self.current_state = new_state
            return event
        
        return None
    
    def _calculate_state(self) -> str:
        """Determine current targeting pattern"""
        all_punches = self.buffer.get_all()
        
        # Need minimum punches
        if len(all_punches) < self.min_punches:
            return "MIXED"
        
        # Count targets (only landed punches)
        landed_punches = [p for p in all_punches if p.outcome == 'landed']
        
        if not landed_punches:
            return "MIXED"
        
        head_count = sum(1 for p in landed_punches if p.target == 'head')
        body_count = sum(1 for p in landed_punches if p.target == 'body')
        total = head_count + body_count
        
        if total == 0:
            return "MIXED"
        
        # Calculate percentages
        head_pct = head_count / total
        body_pct = body_count / total
        
        # Determine state
        if body_pct >= self.focus_threshold:
            return "BODY_FOCUS"
        elif head_pct >= self.focus_threshold:
            return "HEAD_FOCUS"
        else:
            return "MIXED"
    
    def _create_event(self, new_state: str) -> Event:
        """Create target zone shift event"""
        landed = [p for p in self.buffer.get_all() if p.outcome == 'landed']
        head_count = sum(1 for p in landed if p.target == 'head')
        body_count = sum(1 for p in landed if p.target == 'body')
        
        return Event(
            type="target_zone_shift",
            priority=self.priorities[new_state],
            message=random.choice(self.contexts[new_state]),
            context={
                "from_state": self.current_state,
                "to_state": new_state,
                "head_count": head_count,
                "body_count": body_count
            }
        )