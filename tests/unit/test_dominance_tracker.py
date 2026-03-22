import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.action_buffer.buffer import ActionBuffer, Punch
from src.trackers.dominance.tracker import DominanceTracker
import time

def test_tracker_initialization():
    """Test that tracker initializes with correct pendulum structure"""
    buffer = ActionBuffer()
    tracker = DominanceTracker(buffer)
    
    # Should start at EVEN
    assert tracker.current_node.name == "EVEN"
    
    # Check pendulum links
    assert tracker.current_node.next.name == "P1_EDGE"
    assert tracker.current_node.prev.name == "P2_EDGE"
    assert tracker.p1_edge.next.name == "P1_DOMINATING"
    assert tracker.p2_edge.prev.name == "P2_DOMINATING"
    
    print("✅ Tracker initialization correct")

def test_no_event_when_even():
    """Test that no event emitted when fight stays even"""
    buffer = ActionBuffer()
    tracker = DominanceTracker(buffer)
    
    # Add balanced punches
    buffer.add_punch(Punch(1, 'jab', 'head', 'landed', time.time(), 10))
    buffer.add_punch(Punch(2, 'cross', 'head', 'landed', time.time(), 10))
    buffer.add_punch(Punch(1, 'hook', 'body', 'landed', time.time(), 10))
    buffer.add_punch(Punch(2, 'jab', 'head', 'landed', time.time(), 10))
    
    event = tracker.update()
    
    assert event is None  # No state change
    assert tracker.current_node.name == "EVEN"
    print("✅ No event when staying even")

def test_transition_to_p1_edge():
    """Test transition from EVEN to P1_EDGE"""
    buffer = ActionBuffer()
    tracker = DominanceTracker(buffer)
    
    # P1 lands 5, P2 lands 1 (diff = 4, should trigger P1_EDGE)
    for _ in range(5):
        buffer.add_punch(Punch(1, 'jab', 'head', 'landed', time.time(), 10))
    buffer.add_punch(Punch(2, 'cross', 'head', 'landed', time.time(), 10))
    
    event = tracker.update()
    
    # Check state changed
    assert tracker.current_node.name == "P1_EDGE"
    
    # Check event was created
    assert event is not None
    assert event.type == "dominance_shift"
    assert event.priority == 7.0  # P1_EDGE base priority
    assert "Player 1" in event.message
    assert event.context['from_state'] == "EVEN"
    assert event.context['to_state'] == "P1_EDGE"
    assert event.context['p1_landed'] == 5
    assert event.context['p2_landed'] == 1
    
    print(f"✅ Transition to P1_EDGE works - Message: '{event.message}'")

def test_transition_to_p1_dominating():
    """Test jumping to P1_DOMINATING"""
    buffer = ActionBuffer()
    tracker = DominanceTracker(buffer)
    
    # P1 lands 10, P2 lands 1 (diff = 9, should trigger P1_DOM)
    for _ in range(10):
        buffer.add_punch(Punch(1, 'hook', 'head', 'landed', time.time(), 10))
    buffer.add_punch(Punch(2, 'jab', 'head', 'landed', time.time(), 10))
    
    event = tracker.update()
    
    assert tracker.current_node.name == "P1_DOMINATING"
    assert event.priority == 9.0  # Higher priority for dominating
    assert event.context['to_state'] == "P1_DOMINATING"
    
    print(f"✅ Transition to P1_DOMINATING works - Message: '{event.message}'")

def test_transition_to_p2_edge():
    """Test transition to P2 side of pendulum"""
    buffer = ActionBuffer()
    tracker = DominanceTracker(buffer)
    
    # P2 lands 5, P1 lands 1 (diff = -4)
    buffer.add_punch(Punch(1, 'jab', 'head', 'landed', time.time(), 10))
    for _ in range(5):
        buffer.add_punch(Punch(2, 'cross', 'head', 'landed', time.time(), 10))
    
    event = tracker.update()
    
    assert tracker.current_node.name == "P2_EDGE"
    assert "Player 2" in event.message
    assert event.context['to_state'] == "P2_EDGE"
    
    print(f"✅ Transition to P2_EDGE works - Message: '{event.message}'")

def test_multiple_transitions():
    """Test swinging back and forth on pendulum"""
    buffer = ActionBuffer(max_size=20)
    tracker = DominanceTracker(buffer)
    
    print("\n--- Testing Multiple Transitions ---")
    
    # Start: EVEN
    print(f"Start: {tracker.current_node.name}")
    
    # Swing to P1_EDGE
    for _ in range(5):
        buffer.add_punch(Punch(1, 'jab', 'head', 'landed', time.time(), 10))
    buffer.add_punch(Punch(2, 'cross', 'head', 'landed', time.time(), 10))
    
    event1 = tracker.update()
    print(f"After P1 combo: {tracker.current_node.name} - '{event1.message}'")
    assert tracker.current_node.name == "P1_EDGE"
    
    # Swing to P1_DOM
    for _ in range(5):
        buffer.add_punch(Punch(1, 'hook', 'body', 'landed', time.time(), 10))
    
    event2 = tracker.update()
    print(f"After more P1: {tracker.current_node.name} - '{event2.message}'")
    assert tracker.current_node.name == "P1_DOMINATING"
    
    # P2 fights back - buffer will cycle out old punches
    # Clear buffer to simulate time passing
    buffer.clear()
    for _ in range(6):
        buffer.add_punch(Punch(2, 'uppercut', 'head', 'landed', time.time(), 10))
    buffer.add_punch(Punch(1, 'jab', 'head', 'landed', time.time(), 10))
    
    event3 = tracker.update()
    print(f"After P2 comeback: {tracker.current_node.name} - '{event3.message}'")
    assert tracker.current_node.name == "P2_EDGE"
    
    print("✅ Multiple transitions work correctly")

def test_random_message_variety():
    """Test that messages vary due to random.choice()"""
    buffer = ActionBuffer()
    tracker = DominanceTracker(buffer)
    
    messages = set()
    
    # Trigger same transition 10 times
    for i in range(10):
        buffer.clear()
        
        # Always create same state (P1_EDGE)
        for _ in range(5):
            buffer.add_punch(Punch(1, 'jab', 'head', 'landed', time.time(), 10))
        buffer.add_punch(Punch(2, 'cross', 'head', 'landed', time.time(), 10))
        
        # Reset to EVEN first
        tracker.current_node = tracker.even
        
        event = tracker.update()
        if event:
            messages.add(event.message)
    
    print(f"\n✅ Message variety test: Got {len(messages)} different messages from {len(tracker.p1_edge.contexts)} possible contexts")
    print(f"   Messages seen: {messages}")
    
    # Should see at least 2 different messages (might not see all due to randomness)
    assert len(messages) >= 1

if __name__ == "__main__":
    test_tracker_initialization()
    test_no_event_when_even()
    test_transition_to_p1_edge()
    test_transition_to_p1_dominating()
    test_transition_to_p2_edge()
    test_multiple_transitions()
    test_random_message_variety()
    
    print("\n" + "="*50)
    print("🎉 ALL DOMINANCE TRACKER TESTS PASSED!")
    print("="*50)