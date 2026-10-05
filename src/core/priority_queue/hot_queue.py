import time
from src.core import clock
import math
from typing import List, Optional
from src.core.events import Event
from src.core.cooldown.manager import CooldownManager

class PriorityQueue:
    """
    Hot queue for high-priority events.
    Manages event scoring, sorting, and staleness.
    """
    
    def __init__(self, max_size: int = 10, stale_threshold: float = 30.0):
        self.queue: List[Event] = []
        self.max_size = max_size
        self.stale_threshold = stale_threshold  # seconds
        self.recency_half_life = 5.0  # seconds for recency decay
        self.cooldown_manager = CooldownManager()
        
    def push(self, event: Event) -> bool:
        """
        Add event to queue.
        Returns True if added, False if rejected (by cooldown or size).
        """
        # CHECK COOLDOWN FIRST
        if not self.cooldown_manager.check(event):
            # Blocked by cooldown
            return False
        
        # Calculate initial score
        score = self._calculate_score(event)
        
        # Add score to event for tracking
        event.score = score
        
        # Add to queue
        self.queue.append(event)
        
        # Sort by score (highest first)
        self.queue.sort(key=lambda e: e.score, reverse=True)
        
        # Trim to max size (drop lowest priority)
        if len(self.queue) > self.max_size:
            self.queue = self.queue[:self.max_size]
            return score >= self.queue[-1].score
        
        return True
    
    def pop(self, n: int = 1) -> List[Event]:
        """
        Get and remove top N events.
        ACTIVATES COOLDOWNS for popped events.
        """
        if not self.queue:
            return []
        
        # Rescore all events
        self._rescore_all()
        
        # Sort again
        self.queue.sort(key=lambda e: e.score, reverse=True)
        
        # Pop top N
        result = self.queue[:n]
        self.queue = self.queue[n:]
        
        # ACTIVATE COOLDOWNS for events we're using
        for event in result:
            self.cooldown_manager.activate(event)
        
        return result
    
    def peek(self, n: int = 1) -> List[Event]:
        """
        Get top N events WITHOUT removing them.
        """
        if not self.queue:
            return []
        
        # Rescore
        self._rescore_all()
        
        # Sort
        self.queue.sort(key=lambda e: e.score, reverse=True)
        
        return self.queue[:n]
    
    def tick(self):
        """
        Called every cycle - remove stale events.
        """
        current_time = clock.now()
        
        # Remove events older than threshold
        self.queue = [
            e for e in self.queue 
            if (current_time - e.timestamp) < self.stale_threshold
        ]
    
    def size(self) -> int:
        """Current queue size"""
        return len(self.queue)
    
    def clear(self):
        """Empty the queue"""
        self.queue.clear()
    
    def _calculate_score(self, event: Event) -> float:
        """
        Calculate event priority score.
        Score = base_priority × recency_multiplier
        """
        age = clock.now() - event.timestamp
        
        # Recency multiplier: 2^(-age / half_life)
        # Fresh event (age=0): multiplier = 1.0
        # Half-life age: multiplier = 0.5
        # 2× half-life: multiplier = 0.25
        recency_multiplier = math.pow(2, -age / self.recency_half_life)
        
        return event.priority * recency_multiplier
    
    def _rescore_all(self):
        """Recalculate scores for all events"""
        for event in self.queue:
            event.score = self._calculate_score(event)