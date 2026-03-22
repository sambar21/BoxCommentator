from typing import List, Optional
from src.core.events import Event

class ContradictionDetector:
    
    def __init__(self):
        self.last_contradiction = None
    
    def detect(self, tracker_states: dict) -> Optional[Event]:
        
        dominance = tracker_states.get("dominance", "EVEN")
        momentum = tracker_states.get("momentum", "STABLE")
        pace = tracker_states.get("pace", "MODERATE")
        excitement = tracker_states.get("excitement", "MODERATE")
        
        contradiction = self._check_dominance_momentum(dominance, momentum)
        if contradiction:
            return contradiction
        
        contradiction = self._check_pace_excitement(pace, excitement)
        if contradiction:
            return contradiction
        
        return None
    
    def _check_dominance_momentum(self, dominance: str, momentum: str) -> Optional[Event]:
        
        if dominance == "P1_DOMINATING" and momentum == "P2_RISING":
            message = "Player 1 dominating but Player 2 building a comeback"
            if message != self.last_contradiction:
                self.last_contradiction = message
                return Event(
                    type="narrative_tension",
                    priority=8.0,
                    message=message,
                    context={"contradiction_type": "comeback_brewing"}
                )
        
        if dominance == "P2_DOMINATING" and momentum == "P1_RISING":
            message = "Player 2 dominating but Player 1 building a comeback"
            if message != self.last_contradiction:
                self.last_contradiction = message
                return Event(
                    type="narrative_tension",
                    priority=8.0,
                    message=message,
                    context={"contradiction_type": "comeback_brewing"}
                )
        
        return None
    
    def _check_pace_excitement(self, pace: str, excitement: str) -> Optional[Event]:
        
        if pace == "FAST" and excitement == "LOW":
            message = "High pace but ineffective, busy but not landing clean"
            if message != self.last_contradiction:
                self.last_contradiction = message
                return Event(
                    type="narrative_tension",
                    priority=5.5,
                    message=message,
                    context={"contradiction_type": "busy_ineffective"}
                )
        
        return None