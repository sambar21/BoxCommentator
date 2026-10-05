"""
Queue Consumer - Routes events to Track A or Track B based on priority.
"""

from typing import List, Dict, Optional
from src.core.events import Event


class QueueConsumer:
    """
    Consumes events from priority queue and routes to appropriate track.
    
    Track B (urgent): priority >= 9.0
    Track A (analytical): priority < 9.0
    """
    
    def __init__(self, track_b_threshold: float = 9.0):
        self.track_b_threshold = track_b_threshold
        self.never_demote_above = 9.5

        # Content balance tracking (60/40 rule)
        self.track_a_count = 0
        self.track_b_count = 0
        self.window_size = 20  # Rolling window
        
        # Stats
        self.total_consumed = 0
        self.track_a_total = 0
        self.track_b_total = 0
    
    def consume(self, events: List[Event]) -> Optional[Dict]:
        """
        Consume events and decide routing.
        
        Args:
            events: Top events from priority queue (already sorted)
            
        Returns:
            Routing decision dict or None if no events
        """
        if not events:
            return None
        
        self.total_consumed += 1
        
        # Get highest priority event
        primary = events[0]
        supporting = events[1:] if len(events) > 1 else []
        
        # Decide track based on priority
        base_track = self._decide_track(primary.priority)
        
        # Apply 60/40 balance adjustment
        final_track = self._apply_balance(base_track, primary.priority)
        
        # Update counters
        if final_track == 'B':
            self.track_b_count += 1
            self.track_b_total += 1
        else:
            self.track_a_count += 1
            self.track_a_total += 1
        
        # Maintain rolling window
        if (self.track_a_count + self.track_b_count) > self.window_size:
            # Simple reset when window full
            self.track_a_count = int(self.track_a_count * 0.5)
            self.track_b_count = int(self.track_b_count * 0.5)
        
        return {
            'track': final_track,
            'primary_event': primary,
            'supporting_events': supporting,
            'all_events': events
        }
    
    def _decide_track(self, priority: float) -> str:
        """Decide track based on priority threshold"""
        if priority >= self.track_b_threshold:
            return 'B'
        return 'A'
    
    def _apply_balance(self, base_track: str, priority: float = 0.0) -> str:
        """
        Apply 60/40 balance rule.
        Target: 60% Track A, 40% Track B
        Events above 9.5 (knockdowns) are never demoted.
        """
        if base_track == 'B' and priority > self.never_demote_above:
            return 'B'

        total = self.track_a_count + self.track_b_count
        
        if total < 5:
            # Not enough data, use base decision
            return base_track
        
        track_b_ratio = self.track_b_count / total
        
        # If Track B is over 45%, force some events to Track A
        if track_b_ratio > 0.45 and base_track == 'B':
            # Demote to Track A unless priority is REALLY high (>9.5)
            return 'A'
        
        # If Track A is over 65%, boost some events to Track B
        if track_b_ratio < 0.35 and base_track == 'A':
            # Promote to Track B
            return 'B'
        
        return base_track
    
    def get_balance_ratio(self) -> Dict[str, float]:
        """Get current Track A/B balance"""
        total = self.track_a_count + self.track_b_count
        
        if total == 0:
            return {'track_a': 0.5, 'track_b': 0.5}
        
        return {
            'track_a': self.track_a_count / total,
            'track_b': self.track_b_count / total
        }
    
    def get_stats(self) -> Dict:
        """Get consumer statistics"""
        total = self.track_a_total + self.track_b_total
        
        return {
            'total_consumed': self.total_consumed,
            'track_a_total': self.track_a_total,
            'track_b_total': self.track_b_total,
            'track_a_ratio': self.track_a_total / total if total > 0 else 0.0,
            'track_b_ratio': self.track_b_total / total if total > 0 else 0.0,
            'current_balance': self.get_balance_ratio()
        }