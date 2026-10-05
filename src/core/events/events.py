from dataclasses import dataclass
from typing import Dict, Any
import time
from src.core import clock

@dataclass
class Event:
    """
    Represents a fight event that trackers emit.
    
    Events flow: Tracker → Priority Queue → Commentary Generator
    """
    type: str                    # "dominance_shift", "pace_change", etc.
    priority: float              # 0-10 scale (higher = more urgent)
    message: str                 # Human-readable description
    context: Dict[str, Any]      # Additional data about the event
    timestamp: float = None      # When event happened
    score: float = 0.0           # Computed priority score (set by PriorityQueue)
    
    def __post_init__(self):
        """Auto-set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = clock.now()
    
    def __str__(self):
        """Nice string representation for debugging"""
        return f"Event({self.type}, priority={self.priority:.1f}, msg='{self.message}')"