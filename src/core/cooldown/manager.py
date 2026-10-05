"""
Cooldown manager - prevents repetitive commentary.
"""

import time
from src.core import clock
from typing import Dict, Optional
from src.core.events import Event

class CooldownManager:
    """
    Manages cooldowns for event types.
    Prevents talking about the same topic too frequently.
    """
    
    def __init__(self):
        # Base cooldown durations (seconds)
        self.base_cooldowns = {
            'dominance_shift': 10.0,
            'pace_change': 8.0,
            'momentum_reversal': 5.0,
            'excitement_peak': 12.0,
            'target_zone_shift': 10.0,
            'round_change': 15.0,
            'idle_timeout': 3.0,  # Can repeat idle commentary more often
            'knockdown': 2.0
        }
        
        # Priority thresholds for bypassing cooldown
        self.override_thresholds = {
            'dominance_shift': 9.5,
            'pace_change': 9.0,
            'momentum_reversal': 9.0,
            'excitement_peak': 9.5,
            'target_zone_shift': 9.0,
            'round_change': 10.0,  # Never override round changes
            'knockdown': 9.5       # Back-to-back knockdowns are always called
        }
        
        # Active cooldowns: {event_type: expiry_timestamp}
        self.active_cooldowns: Dict[str, float] = {}
        
        # Stats
        self.total_checks = 0
        self.total_blocks = 0
    
    def check(self, event: Event) -> bool:
        """
        Check if event is allowed (not on cooldown).
        
        Returns:
            True if event is allowed
            False if blocked by cooldown
        """
        self.total_checks += 1
        
        event_type = event.type
        current_time = clock.now()
        
        # Check if this event type has an active cooldown
        if event_type in self.active_cooldowns:
            expiry_time = self.active_cooldowns[event_type]
            
            # Is cooldown still active?
            if current_time < expiry_time:
                # Check for priority override
                override_threshold = self.override_thresholds.get(event_type, 999.0)
                
                if event.priority >= override_threshold:
                    # High priority - bypass cooldown
                    return True
                else:
                    # Blocked by cooldown
                    self.total_blocks += 1
                    return False
        
        # No active cooldown or cooldown expired - allow event
        return True
    
    def activate(self, event: Event):
        """
        Activate cooldown for this event type.
        Call this AFTER an event is used for commentary.
        """
        event_type = event.type
        duration = self.base_cooldowns.get(event_type, 5.0)  # Default 5s
        expiry_time = clock.now() + duration
        
        self.active_cooldowns[event_type] = expiry_time
    
    def get_remaining(self, event_type: str) -> float:
        """
        Get remaining cooldown time for an event type.
        Returns 0 if no active cooldown.
        """
        if event_type not in self.active_cooldowns:
            return 0.0
        
        expiry_time = self.active_cooldowns[event_type]
        remaining = max(0.0, expiry_time - clock.now())
        
        # Clean up expired cooldowns
        if remaining == 0.0:
            del self.active_cooldowns[event_type]
        
        return remaining
    
    def clear(self):
        """Clear all active cooldowns"""
        self.active_cooldowns.clear()
    
    def get_stats(self) -> dict:
        """Get cooldown statistics"""
        return {
            'total_checks': self.total_checks,
            'total_blocks': self.total_blocks,
            'block_rate': self.total_blocks / self.total_checks if self.total_checks > 0 else 0.0,
            'active_cooldowns': len(self.active_cooldowns)
        }