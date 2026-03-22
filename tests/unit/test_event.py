from src.core.events import Event
import time

def test_event_creation():
    """Test creating an event"""
    event = Event(
        type="dominance_shift",
        priority=7.5,
        message="Player 1 taking control",
        context={"from": "EVEN", "to": "P1_EDGE"}
    )
    
    assert event.type == "dominance_shift"
    assert event.priority == 7.5
    assert event.message == "Player 1 taking control"
    assert event.context["from"] == "EVEN"
    assert event.timestamp is not None  # Auto-set
    print("✅ Event creation works")

def test_event_with_timestamp():
    """Test event with manual timestamp"""
    custom_time = time.time() - 100  # 100 seconds ago
    
    event = Event(
        type="big_combo",
        priority=9.0,
        message="Huge combination!",
        context={},
        timestamp=custom_time
    )
    
    assert event.timestamp == custom_time
    print("✅ Custom timestamp works")

def test_event_string_representation():
    """Test that events print nicely"""
    event = Event(
        type="test",
        priority=5.0,
        message="Test message",
        context={}
    )
    
    print(f"Event string: {event}")
    assert "test" in str(event)
    assert "5.0" in str(event)
    print("✅ Event string representation works")

if __name__ == "__main__":
    test_event_creation()
    test_event_with_timestamp()
    test_event_string_representation()
    print("\n🎉 All event tests passed!")