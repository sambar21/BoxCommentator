
from src.core.action_buffer.buffer import ActionBuffer, Punch
from src.trackers.pace.tracker import PaceTracker
import time

def test_pace_starts_moderate():
    """Test initial state"""
    buffer = ActionBuffer()
    tracker = PaceTracker(buffer)
    assert tracker.current_state == "MODERATE"
    print("✅ Pace starts moderate")

def test_slow_pace_detection():
    """Test detection of slow pace"""
    buffer = ActionBuffer()
    tracker = PaceTracker(buffer)
    
    # Add only 3 punches in 10 seconds (slow)
    current = time.time()
    buffer.add_punch(Punch(1, 'jab', 'head', 'landed', current - 5, 10))
    buffer.add_punch(Punch(2, 'cross', 'head', 'landed', current - 3, 10))
    buffer.add_punch(Punch(1, 'hook', 'body', 'landed', current, 10))
    
    event = tracker.update()
    
    assert tracker.current_state == "SLOW"
    assert event is not None
    assert event.type == "pace_change"
    assert "tactical" in event.message.lower() or "patient" in event.message.lower() or "slow" in event.message.lower()
    print(f"✅ Slow pace detected - Message: '{event.message}'")

def test_fast_pace_detection():
    """Test detection of fast pace"""
    buffer = ActionBuffer()
    tracker = PaceTracker(buffer)
    
    # Add 25 punches in last 10 seconds (fast!)
    current = time.time()
    for i in range(25):
        player = 1 if i % 2 == 0 else 2
        buffer.add_punch(Punch(player, 'jab', 'head', 'landed', current - (10 - i*0.4), 10))
    
    event = tracker.update()
    
    assert tracker.current_state == "FAST"
    assert event.priority == 6.0  # High priority for fast pace
    # Check any of the FAST state messages from PaceTracker.contexts["FAST"]
    fast_keywords = ["fast", "tempo", "high", "action", "letting", "hands"]
    assert any(kw in event.message.lower() for kw in fast_keywords)
    print(f"✅ Fast pace detected - Message: '{event.message}'")

def test_pace_transitions():
    """Test transitioning between pace states"""
    buffer = ActionBuffer()
    tracker = PaceTracker(buffer)
    
    print("\n--- Testing Pace Transitions ---")
    
    # Start moderate
    current = time.time()
    for i in range(10):
        buffer.add_punch(Punch(1, 'jab', 'head', 'landed', current - (10 - i), 10))
    
    event1 = tracker.update()
    print(f"Start: {tracker.current_state}")
    
    # Speed up to fast
    for i in range(15):
        buffer.add_punch(Punch(2, 'cross', 'head', 'landed', current - (5 - i*0.3), 10))
    
    event2 = tracker.update()
    if event2:
        print(f"After speedup: {tracker.current_state} - '{event2.message}'")
        assert tracker.current_state == "FAST"
    
    # Slow down
    buffer.clear()
    buffer.add_punch(Punch(1, 'jab', 'head', 'landed', current, 10))
    buffer.add_punch(Punch(2, 'cross', 'head', 'landed', current - 1, 10))
    
    event3 = tracker.update()
    if event3:
        print(f"After slowdown: {tracker.current_state} - '{event3.message}'")
        assert tracker.current_state == "SLOW"
    
    print("✅ Pace transitions work")

def test_no_event_when_same_pace():
    """Test no event when pace unchanged"""
    buffer = ActionBuffer()
    tracker = PaceTracker(buffer)
    
    current = time.time()
    # Add moderate amount (stays moderate)
    for i in range(12):
        buffer.add_punch(Punch(1, 'jab', 'head', 'landed', current - (10 - i), 10))
    
    event = tracker.update()
    assert event is None  # No change
    print("✅ No event when pace unchanged")

if __name__ == "__main__":
    test_pace_starts_moderate()
    test_slow_pace_detection()
    test_fast_pace_detection()
    test_pace_transitions()
    test_no_event_when_same_pace()
    
    print("\n" + "="*50)
    print("🎉 ALL PACE TRACKER TESTS PASSED!")
    print("="*50)