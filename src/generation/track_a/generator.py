"""
Track A Generator - Analytical, conversational commentary.
Streaming with interrupt capability.
"""

from typing import Generator, Optional
from src.generation.llm_interface.base_client import BaseLLMClient


class TrackAGenerator:
    """
    Track A: The Analyst
    - Focus: Strategy, patterns, technique
    - Pace: Conversational, thoughtful
    - Generation: Streaming (300ms)
    - Interruptible: Yes
    """

    def __init__(self, llm_client: BaseLLMClient):
        self.llm = llm_client
        self.interrupt_flag = False
        
        # Track A settings
        self.max_tokens = 40  # ~2-3 sentences
        self.temperature = 0.7
    
    def generate_streaming(self, context: dict) -> Generator[str, None, None]:
        """
        Generate commentary with streaming.
        Checks interrupt flag between chunks.
        
        Yields:
            Text chunks
        """
        self.interrupt_flag = False
        
        prompt = self._build_prompt(context)
        
        accumulated = ""
        
        for chunk in self.llm.generate_streaming(
            prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        ):
            # Check interrupt flag
            if self.interrupt_flag:
                # Stop streaming, discard accumulated
                return
            
            accumulated += chunk
            yield chunk
        
        # Full generation complete
        return
    
    def interrupt(self):
        """Signal to stop current generation"""
        self.interrupt_flag = True
    
    def _build_prompt(self, context: dict) -> str:
        """Build Track A prompt from context"""
        
        fighter_1 = context['fighter_names'][1]
        fighter_2 = context['fighter_names'][2]
        
        event_focus = context['event_focus']
        tracker_summary = context['tracker_summary']
        action_summary = context['action_summary']
        tone = context['tone']
        
        round_num = context['round_info']['round_num']
        round_phase = context['round_info']['phase']
        
        # Recent commentary for coherence
        recent = context.get('recent_commentary', [])
        recent_text = "\n".join(recent[-2:]) if recent else "None yet"
        
        rag_stats = context.get('rag_fighter_stats', '')
        rag_history = context.get('rag_history', '')
        live_stats = context.get('live_stats', '')

        rag_block = ""
        if rag_stats or rag_history or live_stats:
            parts = []
            if live_stats:
                parts.append(live_stats)
            if rag_stats:
                parts.append(rag_stats)
            if rag_history:
                parts.append(rag_history)
            rag_block = "\n\n" + "\n\n".join(parts)

        prompt = f"""You are a professional boxing analyst providing live commentary.

FIGHTERS:
- {fighter_1} vs {fighter_2}

CURRENT SITUATION:
- Round {round_num} ({round_phase} phase)
- {event_focus}
- Fight State: {tracker_summary}{rag_block}

RECENT COMMENTARY:
{recent_text}

TONE: {tone}

Generate 1 SHORT sentence (10-15 words max) of analytical commentary. Ground it in the stats above when relevant. Do NOT repeat recent commentary."""
        
        return prompt