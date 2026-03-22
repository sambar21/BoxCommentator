"""
Context Builder - Packages events and fight state into narrative context.
"""

from typing import List, Dict, Any, Optional
from src.core.events import Event
from src.core.action_buffer.buffer import ActionBuffer


class ContextBuilder:
    """
    Builds narrative context packages for LLM generation.
    """
    
    def __init__(self):
        # Track last N outputs for narrative coherence
        self.recent_commentary = []
        self.max_history = 3
        
        # Fighter names (set by orchestrator)
        self.fighter_names = {1: "Fighter 1", 2: "Fighter 2"}
    
    def set_fighter_names(self, p1: str, p2: str):
        """Set fighter names for context"""
        self.fighter_names = {1: p1, 2: p2}
    
    def build(
        self,
        routing_decision: Dict,
        tracker_states: Dict[str, str],
        action_buffer: ActionBuffer,
        round_num: int,
        round_elapsed: float
    ) -> Dict[str, Any]:
        """
        Build complete context package for generation.
        
        Args:
            routing_decision: Output from QueueConsumer
            tracker_states: Current state of all trackers
            action_buffer: Recent fight actions
            round_num: Current round number
            round_elapsed: Seconds into current round
            
        Returns:
            Context package dict
        """
        primary_event = routing_decision['primary_event']
        supporting_events = routing_decision['supporting_events']
        track = routing_decision['track']
        
        # Get recent actions (last 5 seconds)
        recent_actions = action_buffer.get_recent_seconds(5.0)
        
        # Build action summary
        action_summary = self._summarize_actions(recent_actions)
        
        # Build event focus
        event_focus = self._build_event_focus(primary_event, supporting_events)
        
        # Build tracker summary
        tracker_summary = self._build_tracker_summary(tracker_states)
        
        # Determine tone based on track and excitement
        tone = self._determine_tone(track, tracker_states.get('excitement', 'MODERATE'))
        
        # Package everything
        context = {
            'track': track,
            'primary_event': {
                'type': primary_event.type,
                'priority': primary_event.priority,
                'message': primary_event.message,
                'context': primary_event.context
            },
            'supporting_events': [
                {
                    'type': e.type,
                    'priority': e.priority,
                    'message': e.message
                }
                for e in supporting_events
            ],
            'event_focus': event_focus,
            'action_summary': action_summary,
            'tracker_states': tracker_states,
            'tracker_summary': tracker_summary,
            'round_info': {
                'round_num': round_num,
                'elapsed_seconds': round_elapsed,
                'phase': self._get_round_phase(round_elapsed)
            },
            'fighter_names': self.fighter_names,
            'recent_commentary': self.recent_commentary[-self.max_history:],
            'tone': tone
        }
        
        return context
    
    def add_commentary(self, commentary: str):
        """Record generated commentary for narrative coherence"""
        self.recent_commentary.append(commentary)
        
        # Keep only recent history
        if len(self.recent_commentary) > self.max_history:
            self.recent_commentary = self.recent_commentary[-self.max_history:]
    
    def _summarize_actions(self, actions: List) -> Dict[str, Any]:
        """Summarize recent punch activity"""
        if not actions:
            return {
                'total_punches': 0,
                'p1_punches': 0,
                'p2_punches': 0,
                'p1_landed': 0,
                'p2_landed': 0,
                'dominant_punch_type': None
            }
        
        p1_punches = [a for a in actions if a.attacker == 1]
        p2_punches = [a for a in actions if a.attacker == 2]
        
        p1_landed = [a for a in p1_punches if a.outcome == 'landed']
        p2_landed = [a for a in p2_punches if a.outcome == 'landed']
        
        # Count punch types
        punch_types = {}
        for action in actions:
            punch_type = action.punch_type
            punch_types[punch_type] = punch_types.get(punch_type, 0) + 1
        
        dominant_type = max(punch_types.items(), key=lambda x: x[1])[0] if punch_types else None
        
        return {
            'total_punches': len(actions),
            'p1_punches': len(p1_punches),
            'p2_punches': len(p2_punches),
            'p1_landed': len(p1_landed),
            'p2_landed': len(p2_landed),
            'dominant_punch_type': dominant_type,
            'punch_distribution': punch_types
        }
    
    def _build_event_focus(self, primary: Event, supporting: List[Event]) -> str:
        """Build focused narrative from events"""
        focus_parts = [primary.message]
        
        # Add supporting context if relevant
        for event in supporting[:2]:  # Max 2 supporting events
            if event.type != primary.type:  # Avoid redundancy
                focus_parts.append(event.message)
        
        return " | ".join(focus_parts)
    
    def _build_tracker_summary(self, states: Dict[str, str]) -> str:
        """Build human-readable tracker state summary"""
        parts = []
        
        if states.get('dominance') != 'EVEN':
            parts.append(f"Dominance: {states['dominance']}")
        
        if states.get('pace') != 'MODERATE':
            parts.append(f"Pace: {states['pace']}")
        
        if states.get('momentum') != 'STABLE':
            parts.append(f"Momentum: {states['momentum']}")
        
        if states.get('excitement') == 'PEAK':
            parts.append("HIGH INTENSITY")
        
        return " | ".join(parts) if parts else "Balanced fight"
    
    def _determine_tone(self, track: str, excitement: str) -> str:
        """Determine appropriate tone for generation"""
        if track == 'B':
            return 'urgent'
        
        if excitement == 'PEAK':
            return 'energetic'
        elif excitement == 'LULL':
            return 'analytical'
        else:
            return 'conversational'
    
    def _get_round_phase(self, elapsed: float) -> str:
        """Determine phase of round (early/mid/late)"""
        if elapsed < 60:
            return 'early'
        elif elapsed < 120:
            return 'mid'
        else:
            return 'late'