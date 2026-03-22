import random
from typing import Optional, List
from src.core.action_buffer.buffer import ActionBuffer
from src.core.events import Event
from .state_node import StateNode
class DominanceTracker:
    def __init__(self, buffer: ActionBuffer):
        self.buffer = buffer
        
        # Build the pendulum (linked list)
        self.p2_dom = StateNode("P2_DOMINATING", 9.0, [
            "Player 2 is completely controlling this fight",
            "Player 2 dominating right now"
        ])
        
        self.p2_edge = StateNode("P2_EDGE", 7.0, [
            "Player 2 has the edge",
            "Player 2 with a slight advantage"
        ])
        
        self.even = StateNode("EVEN", 4.0, [
            "Dead even right now",
            "This is a tactical battle"
        ])
        
        self.p1_edge = StateNode("P1_EDGE", 7.0, [
            "Player 1 taking control",
            "Player 1 has the edge here"
        ])
        
        self.p1_dom = StateNode("P1_DOMINATING", 9.0, [
            "Player 1 is dominating",
            "Player 1 completely in control"
        ])
        
        # Link them
        self.p2_dom.next = self.p2_edge
        self.p2_edge.prev = self.p2_dom
        self.p2_edge.next = self.even
        self.even.prev = self.p2_edge
        self.even.next = self.p1_edge
        self.p1_edge.prev = self.even
        self.p1_edge.next = self.p1_dom
        self.p1_dom.prev = self.p1_edge
        
        # Start in middle
        self.current_node = self.even
        self.previous_node = None
        
        # Thresholds for movement
        self.edge_threshold = 3
        self.dom_threshold = 8
    
    def update(self) -> Event | None:
        """Check buffer and potentially move along pendulum"""
        
        # Calculate current advantage
        p1_landed = self.buffer.count_landed_by_player(1)
        p2_landed = self.buffer.count_landed_by_player(2)
        diff = p1_landed - p2_landed
        
        # Determine which node we SHOULD be at
        target_node = self._calculate_target_node(diff)
        
        # Did we change nodes?
        if target_node != self.current_node:
            event = self._create_transition_event(target_node)
            self.previous_node = self.current_node
            self.current_node = target_node
            return event
        
        return None
    
    def _calculate_target_node(self, diff: int) -> StateNode:
        """Based on punch difference, which node should we be at?"""
        if diff >= self.dom_threshold:
            return self.p1_dom
        elif diff >= self.edge_threshold:
            return self.p1_edge
        elif diff <= -self.dom_threshold:
            return self.p2_dom
        elif diff <= -self.edge_threshold:
            return self.p2_edge
        else:
            return self.even
    
    def _create_transition_event(self, new_node: StateNode) -> Event:
        """Create event for state transition"""
        return Event(
            type="dominance_shift",
            priority=new_node.priority,
            message=random.choice(new_node.contexts),
            context={
                "from_state": self.current_node.name,
                "to_state": new_node.name,
                "p1_landed": self.buffer.count_landed_by_player(1),
                "p2_landed": self.buffer.count_landed_by_player(2)
            }
        )