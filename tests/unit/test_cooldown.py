import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.cooldown.manager import CooldownManager
from src.core.events import Event
import time

def test_cooldown_allows_first_event():
    """Test first event of a type is allowed"""
    manager = CooldownManager()
    
    event = Event('dominance_shift', 7.0, "Test", {})
    
    assert manager.check(event) == True
    print("✅ First event allowed")

def test_cooldown_blocks_repeat():
    """Test repeat events are blocked"""
    manager = CooldownManager()
    
    event1 = Event('dominance_shift', 7.0, "First", {})
    event2 = Event('dominance_shift', 7.0, "Second", {})
    
    # First allowed
    assert manager.check(event1) == True
    manager.activate(event1)
    
    # Second blocked (cooldown active)
    assert manager.check(event2) == False
    assert manager.total_blocks == 1
    print("✅ Repeat event blocked")

def test_cooldown_expires():
    """Test cooldown expires after duration"""
    manager = CooldownManager()
    manager.base_cooldowns['test_event'] = 0.5  # 0.5 second cooldown
    
    event1 = Event('test_event', 5.0, "First", {})
    event2 = Event('test_event', 5.0, "Second", {})
    
    # First allowed
    assert manager.check(event1) == True
    manager.activate(event1)
    
    # Immediately after - blocked
    assert manager.check(event2) == False
    
    # Wait for cooldown to expire
    time.sleep(0.6)
    
    # Now allowed
    assert manager.check(event2) == True
    print("✅ Cooldown expires correctly")

def test_priority_override():
    """Test high priority bypasses cooldown"""
    manager = CooldownManager()
    
    low_event = Event('dominance_shift', 7.0, "Low priority", {})
    high_event = Event('dominance_shift', 9.8, "High priority", {})
    
    # First event
    assert manager.check(low_event) == True
    manager.activate(low_event)
    
    # Low priority blocked
    low_event2 = Event('dominance_shift', 7.0, "Another low", {})
    assert manager.check(low_event2) == False
    
    # High priority bypasses
    assert manager.check(high_event) == True
    print("✅ Priority override works")

def test_different_types_independent():
    """Test different event types have independent cooldowns"""
    manager = CooldownManager()
    
    dom_event = Event('dominance_shift', 7.0, "Dominance", {})
    pace_event = Event('pace_change', 6.0, "Pace", {})
    
    # Activate dominance cooldown
    assert manager.check(dom_event) == True
    manager.activate(dom_event)
    
    # Pace event still allowed (different type)
    assert manager.check(pace_event) == True
    print("✅ Different event types independent")

if __name__ == "__main__":
    test_cooldown_allows_first_event()
    test_cooldown_blocks_repeat()
    test_cooldown_expires()
    test_priority_override()
    test_different_types_independent()
    
    print("\n" + "="*50)
    print("🎉 ALL COOLDOWN TESTS PASSED!")
    print("="*50)