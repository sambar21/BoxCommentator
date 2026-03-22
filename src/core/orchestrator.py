"""
Commentary Orchestrator - Central coordinator for all fight analysis components.
Processes punches through all trackers and emits events.
"""

import time
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
from src.synthesis.contradiction_detector.detector import ContradictionDetector
from src.generation.track_a.generator import TrackAGenerator
from src.generation.track_b.generator import TrackBGenerator
from src.generation.llm_interface.base_client import BaseLLMClient
from src.generation.llm_interface.llm_factory import LLMFactory


class CommentaryOrchestrator:
    """
    Main orchestrator - coordinates entire commentary pipeline.

    Flow:
    Punch → Buffer → Trackers → Events → Priority Queue →
    Consumer → Context Builder → Track A/B Generator → Commentary
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        # Core components
        self.buffer = ActionBuffer(max_size=20)
        self.priority_queue = PriorityQueue(max_size=10, stale_threshold=30.0)
        self.queue_consumer = QueueConsumer(track_b_threshold=9.0)
        self.context_builder = ContextBuilder()

        # Six trackers
        self.dominance_tracker = DominanceTracker(self.buffer)
        self.pace_tracker = PaceTracker(self.buffer)
        self.momentum_tracker = MomentumTracker(self.buffer)
        self.excitement_tracker = ExcitementTracker(self.buffer)
        self.targets_tracker = TargetZoneTracker(self.buffer)
        self.round_context_tracker = RoundContextTracker()

        # Synthesis
        self.contradiction_detector = ContradictionDetector()

        # LLM and generation
        self.llm = llm_client or LLMFactory.create_client()
        self.track_a_generator = TrackAGenerator(self.llm)
        self.track_b_generator = TrackBGenerator(self.llm)
        
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
        """Set fighter names for commentary"""
        self.context_builder.set_fighter_names(p1, p2)
    
    def start_round(self, round_num: int):
        """Initialize new round"""
        self.current_round = round_num
        self.round_start_time = time.time()
        self.round_context_tracker.set_round(round_num)
        self.buffer.clear()
        self.priority_queue.clear()
    
    def process_punch(self, punch: Punch) -> Optional[str]:
        """
        Process a punch and generate commentary if needed.
        
        This is the main entry point - called every time a punch happens.
        
        Args:
            punch: Punch data
            
        Returns:
            Generated commentary string or None
        """
        # Add to buffer
        self.buffer.add_punch(punch)
        self.stats['total_punches'] += 1
        
        # Update all trackers (they emit events to queue)
        self._process_trackers()
        
        # Tick queue (remove stale events)
        self.priority_queue.tick()
        
        # Decide if we should generate commentary
        return self._generate_commentary_if_needed()
    
    def _process_trackers(self):
        """Update all trackers and push their events to priority queue"""
        
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
        seconds_elapsed = time.time() - self.round_start_time if self.round_start_time else 0
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
            round_elapsed=time.time() - self.round_start_time if self.round_start_time else 0
        )
        
        # Mark Track A as active
        self.track_a_active = True
        self.track_a_context = context['event_focus']
        
        # Generate (streaming, but we'll collect full output for now)
        # TODO: In production, handle streaming with interrupt checking
        full_output = ""
        for chunk in self.track_a_generator.generate_streaming(context):
            full_output += chunk
        
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
            round_elapsed=time.time() - self.round_start_time if self.round_start_time else 0
        )
        
        # Generate with Track A context for coherence
        output = self.track_b_generator.generate(
            context=context,
            track_a_context=self.track_a_context
        )
        
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