from typing import Optional
from src.core.events import Event

class RoundContextTracker:
    
    EARLY_ROUNDS = (1, 2, 3)
    MIDDLE_ROUNDS = (4, 5, 6, 7, 8)
    LATE_ROUNDS = (9, 10, 11, 12)
    
    def __init__(self):
        self.current_round = 1
        self.last_narrative = None
        self.round_narratives = {}
    
    def set_round(self, round_num: int):
        self.current_round = round_num
    
    def analyze_context(self, pace: str, dominance: str, seconds_elapsed: float) -> Optional[Event]:
        
        if self.current_round in self.EARLY_ROUNDS:
            return self._analyze_early_round(pace, dominance, seconds_elapsed)
        elif self.current_round in self.MIDDLE_ROUNDS:
            return self._analyze_middle_round(pace, dominance, seconds_elapsed)
        else:
            return self._analyze_late_round(pace, dominance, seconds_elapsed)
    
    def _analyze_early_round(self, pace: str, dominance: str, seconds_elapsed: float) -> Optional[Event]:
        
        if pace == "FAST" and seconds_elapsed < 30:
            narrative = "Unusual early aggression, someone making a statement"
            if narrative != self.last_narrative:
                self.last_narrative = narrative
                return Event(
                    type="round_context",
                    priority=5.0,
                    message=narrative,
                    context={"round": self.current_round, "pattern": "early_aggression"}
                )
        
        if dominance in ["P1_DOMINATING", "P2_DOMINATING"] and seconds_elapsed < 45:
            narrative = "Quick dominance early, establishing control"
            if narrative != self.last_narrative:
                self.last_narrative = narrative
                return Event(
                    type="round_context",
                    priority=5.5,
                    message=narrative,
                    context={"round": self.current_round, "pattern": "early_dominance"}
                )
        
        return None
    
    def _analyze_middle_round(self, pace: str, dominance: str, seconds_elapsed: float) -> Optional[Event]:
        
        if pace == "SLOW":
            narrative = "Middle rounds turning tactical, feeling each other out"
            if narrative != self.last_narrative:
                self.last_narrative = narrative
                return Event(
                    type="round_context",
                    priority=3.0,
                    message=narrative,
                    context={"round": self.current_round, "pattern": "tactical_middle"}
                )
        
        return None
    
    def _analyze_late_round(self, pace: str, dominance: str, seconds_elapsed: float) -> Optional[Event]:
        
        if pace == "FAST":
            narrative = "Championship rounds, high pace, both digging deep"
            if narrative != self.last_narrative:
                self.last_narrative = narrative
                return Event(
                    type="round_context",
                    priority=6.0,
                    message=narrative,
                    context={"round": self.current_round, "pattern": "late_war"}
                )
        
        if pace == "SLOW":
            narrative = "Late round chess match, conserving energy"
            if narrative != self.last_narrative:
                self.last_narrative = narrative
                return Event(
                    type="round_context",
                    priority=4.0,
                    message=narrative,
                    context={"round": self.current_round, "pattern": "late_tactical"}
                )
        
        return None