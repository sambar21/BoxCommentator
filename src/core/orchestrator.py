"""
Commentary Orchestrator - Central coordinator for all fight analysis components.
Processes punches through all trackers and emits events.
"""

import time
from src.core import clock
from typing import List, Optional, Dict
from src.core.action_buffer.buffer import ActionBuffer, Punch
from src.core.events import Event
from src.core.priority_queue import PriorityQueue
from src.core.queue_consumer.consumer import QueueConsumer
from src.core.context_builder.builder import ContextBuilder
from src.trackers.dominance import DominanceTracker
from src.trackers.pace import PaceTracker
from src.trackers.momentum.tracker import MomentumTracker
from src.trackers.excitement import ExcitementTracker
from src.trackers.targets import TargetZoneTracker
from src.trackers.round_context import RoundContextTracker
from src.trackers.knockdown import KnockdownTracker
from src.synthesis.contradiction_detector.detector import ContradictionDetector
from src.generation.track_a.generator import TrackAGenerator
from src.generation.track_b.generator import TrackBGenerator
from src.generation.llm_interface.base_client import BaseLLMClient
from src.generation.llm_interface.llm_factory import LLMFactory
from src.generation.timing import GenerationRecord, collect_timed

try:
    from src.retrieval.fight_memory.store import FightMemoryStore
    from src.retrieval.historical_search.searcher import HistoricalSearcher
    _RETRIEVAL_AVAILABLE = True
except ImportError:
    _RETRIEVAL_AVAILABLE = False


class CommentaryOrchestrator:
    """
    Main orchestrator - coordinates entire commentary pipeline.

    Flow:
    Punch -> Buffer -> Trackers -> Events -> Priority Queue ->
    Consumer -> Context Builder -> Track A/B Generator -> Commentary
    """

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        use_rag: bool = True,
        track_a_llm: Optional[BaseLLMClient] = None,
        track_b_llm: Optional[BaseLLMClient] = None,
        context_mode: str = "recent",
        history_cap: int = 8,
        keep_prompts: bool = False,
    ):
        """
        llm_client       one backend for both tracks (default: LLMFactory from config)
        track_a_llm/_b_llm  per-track backends for routing; fall back to llm_client
        context_mode     recent | full | capped (see ContextBuilder)
        keep_prompts     store each prompt on its GenerationRecord (for quality checks)
        """
        self.keep_prompts = keep_prompts

        # Core components
        self.buffer = ActionBuffer(max_size=20)
        self.priority_queue = PriorityQueue(max_size=10, stale_threshold=30.0)
        self.queue_consumer = QueueConsumer(track_b_threshold=9.0)
        self.context_builder = ContextBuilder(history_mode=context_mode, history_cap=history_cap)
        self.context_builder.attach_window_analyzer(self.buffer)

        # Six trackers
        self.dominance_tracker = DominanceTracker(self.buffer)
        self.pace_tracker = PaceTracker(self.buffer)
        self.momentum_tracker = MomentumTracker(self.buffer)
        self.excitement_tracker = ExcitementTracker(self.buffer)
        self.targets_tracker = TargetZoneTracker(self.buffer)
        self.round_context_tracker = RoundContextTracker()
        self.knockdown_tracker = KnockdownTracker(
            self.buffer, names=lambda: self.context_builder.fighter_names
        )

        # Synthesis
        self.contradiction_detector = ContradictionDetector()

        # LLM and generation
        if llm_client is None and track_a_llm is None and track_b_llm is None:
            llm_client = LLMFactory.create_client()
        self.llm = llm_client or track_a_llm or track_b_llm
        self.llm_a = track_a_llm or self.llm
        self.llm_b = track_b_llm or self.llm
        self.track_a_generator = TrackAGenerator(self.llm_a)
        self.track_b_generator = TrackBGenerator(self.llm_b)

        # RAG components
        if _RETRIEVAL_AVAILABLE and use_rag:
            self._fight_memory = FightMemoryStore()
            self._history = HistoricalSearcher()
            self.context_builder.attach_retrieval(
                self._fight_memory, self._history, self.buffer
            )
        else:
            self._fight_memory = None
            self._history = None
        
        # Per-generation timing/attribution records (see src/generation/timing.py)
        self.generation_log: List[GenerationRecord] = []

        # Round state
        self.current_round = 1
        self.round_start_time = None
        
        # Track A state (for interrupts)
        self.track_a_active = False
        self.track_a_context = None
        
        # Stats
        self.stats = {
            'total_punches': 0,
            'total_events_emitted': 0,
            'total_events_queued': 0,
            'cooldown_blocks': 0,
            'commentary_generated': 0,
            'track_a_generated': 0,
            'track_b_generated': 0,
            'track_a_interrupts': 0,
            'current_dominance': 'EVEN',
            'current_pace': 'MODERATE',
            'current_momentum': 'STABLE',
            'current_excitement': 'MODERATE',
            'current_targets': 'BALANCED'
        }
    
    def set_fighter_names(self, p1: str, p2: str):
        """Set fighter names and seed RAG store with their profiles."""
        self.context_builder.set_fighter_names(p1, p2)
        if self._fight_memory is not None:
            self._fight_memory.seed_fighters([p1, p2])
        if self._history is not None:
            self._history.seed()
    
    def start_round(self, round_num: int):
        """Initialize new round"""
        self.current_round = round_num
        self.round_start_time = clock.now()
        self.round_context_tracker.set_round(round_num)
        self.buffer.clear()
        self.priority_queue.clear()
    
    def process_punch(self, punch: Punch) -> Optional[str]:
        """
        Process a punch and generate commentary if needed.

        Returns:
            Generated commentary string or None
        """
        t0 = time.perf_counter()

        self.buffer.add_punch(punch)
        self.stats['total_punches'] += 1

        self._process_trackers()
        self.priority_queue.tick()

        result = self._generate_commentary_if_needed()

        latency_ms = (time.perf_counter() - t0) * 1000
        self._record_latency(latency_ms)

        return result

    def _log_generation(self, track: str, context: dict, text: str,
                        ttft_ms, total_ms: float, n_chunks: int):
        """Keep a per-generation record for benchmarks and quality checks."""
        primary = context['primary_event']
        llm = self.llm_a if track == 'A' else self.llm_b
        generator = self.track_a_generator if track == 'A' else self.track_b_generator
        self.generation_log.append(GenerationRecord(
            track=track,
            backend=llm.get_provider_name(),
            event_type=primary['type'],
            event_context=primary['context'],
            punch_index=self.stats['total_punches'],
            text=text,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            n_chunks=n_chunks,
            n_tokens=getattr(llm, 'last_completion_tokens', None),
            prompt_tokens=getattr(llm, 'last_prompt_tokens', None),
            cached_tokens=getattr(llm, 'last_cached_tokens', None),
            prompt_chars=len(generator.last_prompt),
            prompt=generator.last_prompt if self.keep_prompts else "",
        ))

    def _record_latency(self, latency_ms: float):
        """Track latency for p95 reporting."""
        samples = self.stats.setdefault('_latency_samples', [])
        samples.append(latency_ms)
        if len(samples) > 1000:
            self.stats['_latency_samples'] = samples[-1000:]

    def get_p95_latency(self) -> float:
        """Return p95 end-to-end latency in milliseconds."""
        samples = self.stats.get('_latency_samples', [])
        if not samples:
            return 0.0
        sorted_s = sorted(samples)
        idx = int(len(sorted_s) * 0.95)
        return sorted_s[min(idx, len(sorted_s) - 1)]
    
    def _process_trackers(self):
        """Update all trackers and push their events to priority queue"""

        # Knockdown (highest priority, checked first)
        knockdown_event = self.knockdown_tracker.update()
        if knockdown_event:
            self._push_event(knockdown_event)

        # Dominance
        dom_event = self.dominance_tracker.update()
        if dom_event:
            self._push_event(dom_event)
            self.stats['current_dominance'] = self.dominance_tracker.current_node.name
        
        # Pace
        pace_event = self.pace_tracker.update()
        if pace_event:
            self._push_event(pace_event)
            self.stats['current_pace'] = self.pace_tracker.current_state
        
        # Momentum
        momentum_event = self.momentum_tracker.update()
        if momentum_event:
            self._push_event(momentum_event)
            self.stats['current_momentum'] = self.momentum_tracker.current_state
        
        # Excitement (includes idle timeout)
        excitement_event = self.excitement_tracker.update()
        if excitement_event:
            self._push_event(excitement_event)
            self.stats['current_excitement'] = self.excitement_tracker.current_state
        
        # Target zones
        targets_event = self.targets_tracker.update()
        if targets_event:
            self._push_event(targets_event)
            self.stats['current_targets'] = self.targets_tracker.current_state
        
        # Round context
        tracker_states = self.get_tracker_states()
        seconds_elapsed = clock.now() - self.round_start_time if self.round_start_time else 0
        round_event = self.round_context_tracker.analyze_context(
            tracker_states['pace'],
            tracker_states['dominance'],
            seconds_elapsed
        )
        if round_event:
            self._push_event(round_event)
        
        # Contradiction detection
        contradiction_event = self.contradiction_detector.detect(tracker_states)
        if contradiction_event:
            self._push_event(contradiction_event)
    
    def _push_event(self, event: Event):
        """Push event to priority queue with cooldown checking"""
        self.stats['total_events_emitted'] += 1
        
        accepted = self.priority_queue.push(event)
        
        if accepted:
            self.stats['total_events_queued'] += 1
        else:
            self.stats['cooldown_blocks'] += 1
    
    def _generate_commentary_if_needed(self) -> Optional[str]:
        """
        Check queue and generate commentary if conditions met.
        
        Generation triggers:
        1. High priority event (>= 9.0) - immediate Track B
        2. Queue has 3+ events - Track A
        3. Idle timeout event - Track A
        
        Returns:
            Generated commentary or None
        """
        # Peek at queue without consuming
        top_events = self.priority_queue.peek(n=3)
        
        if not top_events:
            return None
        
        # Check if highest priority event is urgent (Track B territory)
        highest_priority = top_events[0].priority
        
        # Track B immediate trigger
        if highest_priority >= 9.0:
            return self._generate_track_b()
        
        # Track A triggers
        queue_size = self.priority_queue.size()
        
        # Generate Track A if queue has enough events or idle timeout
        should_generate_track_a = (
            queue_size >= 2 or
            any(e.type == 'idle_timeout' for e in top_events)
        )
        
        if should_generate_track_a:
            return self._generate_track_a()
        
        return None
    
    def _generate_track_a(self) -> str:
        """Generate Track A (analytical) commentary"""
        # Pop events from queue
        events = self.priority_queue.pop(n=3)
        
        if not events:
            return ""
        
        # Route through consumer
        routing = self.queue_consumer.consume(events)
        
        if not routing:
            return ""
        
        # Build context
        context = self.context_builder.build(
            routing_decision=routing,
            tracker_states=self.get_tracker_states(),
            action_buffer=self.buffer,
            round_num=self.current_round,
            round_elapsed=clock.now() - self.round_start_time if self.round_start_time else 0
        )
        
        # Mark Track A as active
        self.track_a_active = True
        self.track_a_context = context['event_focus']
        
        # Generate (streaming, but we'll collect full output for now)
        # TODO: In production, handle streaming with interrupt checking
        full_output, ttft_ms, total_ms, n_chunks = collect_timed(
            self.track_a_generator.generate_streaming(context)
        )
        self._log_generation('A', context, full_output, ttft_ms, total_ms, n_chunks)

        # Mark Track A complete
        self.track_a_active = False
        
        # Update stats
        self.stats['commentary_generated'] += 1
        self.stats['track_a_generated'] += 1
        
        # Record for narrative coherence
        self.context_builder.add_commentary(full_output)
        
        return full_output
    
    def _generate_track_b(self) -> str:
        """Generate Track B (urgent) commentary"""
        # Check if Track A is active - if so, interrupt it
        if self.track_a_active:
            self.track_a_generator.interrupt()
            self.stats['track_a_interrupts'] += 1
        
        # Pop ONLY the urgent event
        events = self.priority_queue.pop(n=1)
        
        if not events:
            return ""
        
        # Route through consumer
        routing = self.queue_consumer.consume(events)
        
        if not routing:
            return ""
        
        # Build context
        context = self.context_builder.build(
            routing_decision=routing,
            tracker_states=self.get_tracker_states(),
            action_buffer=self.buffer,
            round_num=self.current_round,
            round_elapsed=clock.now() - self.round_start_time if self.round_start_time else 0
        )
        
        # Generate with Track A context for coherence
        output, ttft_ms, total_ms, n_chunks = collect_timed(
            self.track_b_generator.generate_streaming(
                context=context,
                track_a_context=self.track_a_context
            )
        )
        self._log_generation('B', context, output, ttft_ms, total_ms, n_chunks)

        # Update stats
        self.stats['commentary_generated'] += 1
        self.stats['track_b_generated'] += 1
        
        # Record for narrative coherence
        self.context_builder.add_commentary(output)
        
        # Clear Track A context
        self.track_a_context = None
        
        return output
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics"""
        queue_stats = {
            'queue_size': self.priority_queue.size(),
        }
        cooldown_stats = self.priority_queue.cooldown_manager.get_stats()
        consumer_stats = self.queue_consumer.get_stats()
        
        return {
            **self.stats,
            **queue_stats,
            **cooldown_stats,
            **consumer_stats
        }
    
    def get_tracker_states(self) -> dict:
        """Get current state of all trackers"""
        return {
            'dominance': self.stats['current_dominance'],
            'pace': self.stats['current_pace'],
            'momentum': self.stats['current_momentum'],
            'excitement': self.stats['current_excitement'],
            'targets': self.stats['current_targets']
        }