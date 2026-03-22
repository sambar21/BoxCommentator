"""
Event Synthesizer - Template-based fallback for when LLM unavailable.
Used for graceful degradation.
"""

from typing import List
from src.core.events import Event


class EventSynthesizer:
    """
    Template-based commentary generation.
    Used as fallback when LLM is unavailable or for emergency mode.
    """
    
    def __init__(self):
        self.fighter_names = {1: "Fighter 1", 2: "Fighter 2"}
    
    def set_fighter_names(self, p1: str, p2: str):
        self.fighter_names = {1: p1, 2: p2}
    
    def synthesize(self, events: List[Event], tracker_states: dict, recent_actions: list = None) -> str:
        """
        Generate template-based commentary from events.
        
        This is a FALLBACK - production uses Track A/B generators with LLM.
        """
        if not events:
            return ""
        
        if len(events) == 1:
            return self._synthesize_single(events[0], tracker_states)
        
        return self._synthesize_multiple(events, tracker_states)
    
    def _synthesize_single(self, event: Event, tracker_states: dict) -> str:
        """Handle single event"""
        event_type = event.type
        
        # Use event's built-in message as base
        base_message = event.message
        
        # Enhance based on event type
        if event_type == 'dominance_shift':
            return self._enhance_dominance(event, tracker_states)
        elif event_type == 'excitement_peak':
            return self._enhance_excitement(event, tracker_states)
        elif event_type == 'momentum_reversal':
            return self._enhance_momentum(event, tracker_states)
        elif event_type == 'idle_timeout':
            return self._enhance_idle(event, tracker_states)
        else:
            return base_message
    
    def _synthesize_multiple(self, events: List[Event], tracker_states: dict) -> str:
        """Handle multiple events - find patterns"""
        event_types = [e.type for e in events]
        
        # Dominance + Target shift
        if "dominance_shift" in event_types and "target_zone_shift" in event_types:
            return self._synthesize_dominance_targeting(events, tracker_states)
        
        # Pace + Dominance
        if "pace_change" in event_types and "dominance_shift" in event_types:
            return self._synthesize_pace_dominance(events, tracker_states)
        
        # Momentum + Dominance (contradiction)
        if "momentum_reversal" in event_types and "dominance_shift" in event_types:
            return self._synthesize_momentum_dominance(events, tracker_states)
        
        # Default: use highest priority event
        return events[0].message
    
    def _enhance_dominance(self, event: Event, states: dict) -> str:
        """Enhance dominance shift commentary"""
        dominance = states.get('dominance', 'EVEN')
        
        if 'P1' in dominance:
            return f"{self.fighter_names[1]} is taking control here"
        elif 'P2' in dominance:
            return f"{self.fighter_names[2]} is taking control here"
        else:
            return "This fight is evening out"
    
    def _enhance_excitement(self, event: Event, states: dict) -> str:
        """Enhance excitement commentary"""
        return "They're going at it now!"
    
    def _enhance_momentum(self, event: Event, states: dict) -> str:
        """Enhance momentum shift commentary"""
        momentum = states.get('momentum', 'STABLE')
        
        if 'P1' in momentum:
            return f"{self.fighter_names[1]} is building momentum"
        elif 'P2' in momentum:
            return f"{self.fighter_names[2]} is building momentum"
        else:
            return "Momentum shift happening here"
    
    def _enhance_idle(self, event: Event, states: dict) -> str:
        """Enhance idle commentary with context"""
        dominance = states.get('dominance', 'EVEN')
        pace = states.get('pace', 'MODERATE')
        
        if dominance != 'EVEN':
            if 'P1' in dominance:
                return f"{self.fighter_names[1]} controlling the pace here"
            else:
                return f"{self.fighter_names[2]} controlling the pace here"
        
        if pace == 'SLOW':
            return "Tactical chess match developing"
        
        return "Good technical boxing on display"
    
    def _synthesize_dominance_targeting(self, events: List[Event], tracker_states: dict) -> str:
        dominance = tracker_states.get("dominance", "EVEN")
        targets = tracker_states.get("targets", "BALANCED")
        
        if "P1" in dominance and "BODY" in targets:
            return f"{self.fighter_names[1]} taking control with body work"
        elif "P1" in dominance and "HEAD" in targets:
            return f"{self.fighter_names[1]} taking control, headhunting now"
        elif "P2" in dominance and "BODY" in targets:
            return f"{self.fighter_names[2]} taking control with body work"
        elif "P2" in dominance and "HEAD" in targets:
            return f"{self.fighter_names[2]} taking control, headhunting now"
        
        return events[0].message
    
    def _synthesize_pace_dominance(self, events: List[Event], tracker_states: dict) -> str:
        pace = tracker_states.get("pace", "MODERATE")
        dominance = tracker_states.get("dominance", "EVEN")
        
        if pace == "FAST" and "P1" in dominance:
            return f"{self.fighter_names[1]} turning up the tempo, taking control"
        elif pace == "FAST" and "P2" in dominance:
            return f"{self.fighter_names[2]} turning up the tempo, taking control"
        elif pace == "SLOW" and dominance == "EVEN":
            return "Tactical chess match, both being patient"
        
        return events[0].message
    
    def _synthesize_momentum_dominance(self, events: List[Event], tracker_states: dict) -> str:
        momentum = tracker_states.get("momentum", "STABLE")
        dominance = tracker_states.get("dominance", "EVEN")
        
        if "P1_RISING" in momentum and "P2" in dominance:
            return f"{self.fighter_names[1]} fighting back despite being dominated"
        elif "P2_RISING" in momentum and "P1" in dominance:
            return f"{self.fighter_names[2]} fighting back despite being dominated"
        
        return events[0].message