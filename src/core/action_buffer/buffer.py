from dataclasses import dataclass
from collections import deque
from typing import List, Optional
import time

@dataclass
class Punch:
    attacker: int        # 1 or 2
    punch_type: str      # jab, cross, hook, uppercut
    target: str          # head or body
    outcome: str         # landed, missed, blocked
    timestamp: float     # time punch happened
    damage: int = 0      # optional damage value
class ActionBuffer:
    """
    Circular buffer storing recent punches.
    Fast access to last N actions.
    """

    def __init__(self, max_size: int = 20):
        self.max_size = max_size
        self.buffer: deque[Punch] = deque(maxlen=max_size)

    def add_punch(self, punch: Punch):
        """Add a punch to the buffer"""
        self.buffer.append(punch)
    
    def get_recent(self, n: int) -> List[Punch]:
        """Get last N punches"""
        items = list(self.buffer)
        return items[-n:] if len(items) >= n else items
    
    def get_recent_seconds(self, seconds: float) -> List[Punch]:
        """Get punches from last N seconds"""
        if not self.buffer:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - seconds
        
        return [p for p in self.buffer if p.timestamp >= cutoff_time]
    
    def count_landed_by_player(self, player_id: int) -> int:
        """Count landed punches by a specific player"""
        return sum(1 for p in self.buffer if p.attacker == player_id and p.outcome == 'landed')
    
    def count_by_player(self, player_id: int) -> int:
        """Count all punches thrown by a specific player"""
        return sum(1 for p in self.buffer if p.attacker == player_id)
    
    def get_punches_by_player(self, player_id: int) -> List[Punch]:
        """Get all punches thrown by a specific player"""
        return [p for p in self.buffer if p.attacker == player_id]
    
    def count_by_type(self, punch_type: str) -> int:
        """Count punches of a specific type"""
        return sum(1 for p in self.buffer if p.punch_type == punch_type)
    
    def count_by_target(self, target: str) -> int:
        """Count punches to a specific target"""
        return sum(1 for p in self.buffer if p.target == target)
    
    def count_by_outcome(self, outcome: str) -> int:
        """Count punches with a specific outcome"""
        return sum(1 for p in self.buffer if p.outcome == outcome)
    
    def get_all(self) -> List[Punch]:
        """Get all punches in buffer"""
        return list(self.buffer)
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)