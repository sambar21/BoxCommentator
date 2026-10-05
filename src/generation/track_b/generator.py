"""
Track B Generator - Urgent, reactive commentary.
Direct generation for speed.
"""

from typing import Optional
from src.generation.llm_interface.base_client import BaseLLMClient


class TrackBGenerator:
    """
    Track B: The Reactor
    Focus: Big punches, knockdowns, hurt fighters
    Pace: Immediate, explosive
    Generation: Direct (200ms)
    Interruptible: No (completes quickly)
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.last_prompt = ""

        # Track B settings
        self.max_tokens = 50  # ~1-2 sentences, SHORT
        self.temperature = 0.9  # More energetic
    
    def generate(self, context: dict, track_a_context: Optional[str] = None) -> str:
        """
        Generate urgent commentary.
        
        Args:
            context: Full context package
            track_a_context: What Track A was discussing (for coherence)
            
        Returns:
            Generated commentary
        """
        prompt = self._build_prompt(context, track_a_context)
        
        return self.llm.generate(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
    
    def generate_streaming(self, context: dict, track_a_context: Optional[str] = None):
        """Same prompt and params as generate(), but streamed so TTFT is measurable."""
        prompt = self._build_prompt(context, track_a_context)
        self.last_prompt = prompt

        yield from self.llm.generate_streaming(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

    def _build_prompt(self, context: dict, track_a_context: Optional[str]) -> str:
        """Build Track B prompt with transition words"""
        
        fighter_1 = context['fighter_names'][1]
        fighter_2 = context['fighter_names'][2]
        
        primary_event = context['primary_event']
        event_message = primary_event['message']
        event_type = primary_event['type']
        
        # Determine transition word based on context
        transition = self._select_transition(event_type, track_a_context)
        
        prompt = f"""You are a boxing commentator calling an URGENT moment.

FIGHTERS: {fighter_1} vs {fighter_2}

URGENT EVENT: {event_message}

"""
        
        if track_a_context:
            prompt += f"""CONTEXT: You were just discussing: "{track_a_context}"

TRANSITION: Use "{transition}" to interrupt naturally.

"""
        
        prompt += """Generate 1 SHORT sentence (5-10 words) with ENERGY and URGENCY. Use ALL CAPS for emphasis on key words."""
        
        return prompt
    
    def _select_transition(self, event_type: str, prior_context: Optional[str]) -> str:
        """Select appropriate transition word"""
        
        # If no prior context, no transition needed
        if not prior_context:
            return "WAIT"
        
        # Reversal transitions
        if event_type in ['momentum_reversal', 'dominance_shift']:
            return "BUT"
        
        # Escalation transitions
        if event_type in ['excitement_peak']:
            return "AND NOW"
        
        # Continuation transitions
        if event_type in ['combo_landed', 'fighter_hurt']:
            return "AND"
        
        # Default
        return "WAIT"