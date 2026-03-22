import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.priority_queue import PriorityQueue
from src.core.events import Event
import time

def test_queue_initialization():
    """Test queue starts empty"""
    queue = PriorityQueue(max_size=5)
    assert queue.size() == 0
    print(" Queue initializes empty")

def test_push_and_pop():
    """Test basic push/pop"""
    queue = PriorityQueue()
    
    event1 = Event("test1", 5.0, "First event", {})
    event2 = Event("test2", 8.0, "Second event", {})
    
    queue.push(event1)
    queue.push(event2)
    
    assert queue.size() == 2
    
    # Pop should return highest priority first
    popped = queue.pop(1)
    assert len(popped) == 1
    assert popped[0].priority == 8.0  # event2 has higher priority
    assert queue.size() == 1
    
    print(" Push and pop work correctly")

def test_priority_ordering():
    """Test events sorted by priority"""
    queue = PriorityQueue()
    
    queue.push(Event("low", 3.0, "Low priority", {}))
    queue.push(Event("high", 9.0, "High priority", {}))
    queue.push(Event("medium", 6.0, "Medium priority", {}))
    
    # Peek at top 3
    top = queue.peek(3)
    
    assert top[0].priority == 9.0  # Highest first
    assert top[1].priority == 6.0  # Medium second
    assert top[2].priority == 3.0  # Lowest third
    
    print("✅ Events sorted by priority")

def test_recency_boost():
    """Test that recent events beat old ones"""
    queue = PriorityQueue()
    
    # Old event with high priority
    old_event = Event("old", 8.0, "Old but high priority", {})
    old_event.timestamp = time.time() - 10  # 10 seconds ago
    queue.push(old_event)
    
    time.sleep(0.1)  # Small delay
    
    # Recent event with lower priority
    new_event = Event("new", 6.0, "Recent but lower priority", {})
    queue.push(new_event)
    
    # The recent one should win due to recency boost
    top = queue.peek(1)
    
    print(f"   Old event score: {old_event.score:.2f}")
    print(f"   New event score: {new_event.score:.2f}")
    
    # New event should have higher score despite lower base priority
    assert top[0].type == "new"
    
    print("✅ Recency boost works")

def test_max_size_enforcement():
    """Test queue doesn't exceed max size"""
    queue = PriorityQueue(max_size=3)
    
    queue.push(Event("e1", 5.0, "Event 1", {}))
    queue.push(Event("e2", 6.0, "Event 2", {}))
    queue.push(Event("e3", 7.0, "Event 3", {}))
    queue.push(Event("e4", 8.0, "Event 4", {}))
    queue.push(Event("e5", 4.0, "Event 5", {}))
    
    # Should only keep 3 highest priority
    assert queue.size() == 3
    
    top = queue.peek(3)
    assert top[0].priority == 8.0
    assert top[1].priority == 7.0
    assert top[2].priority == 6.0
    
    print("✅ Max size enforced")

def test_stale_event_removal():
    """Test old events get pruned"""
    queue = PriorityQueue(stale_threshold=2.0)  # 2 second threshold
    
    # Add old event
    old = Event("old", 5.0, "Old event", {})
    old.timestamp = time.time() - 3  # 3 seconds ago
    queue.push(old)
    
    # Add recent event
    queue.push(Event("new", 5.0, "New event", {}))
    
    assert queue.size() == 2
    
    # Tick to prune stale
    queue.tick()
    
    # Old one should be removed
    assert queue.size() == 1
    assert queue.peek(1)[0].type == "new"
    
    print("✅ Stale events pruned")

def test_peek_vs_pop():
    """Test peek doesn't remove, pop does"""
    queue = PriorityQueue()
    
    queue.push(Event("e1", 5.0, "Event 1", {}))
    queue.push(Event("e2", 6.0, "Event 2", {}))
    
    # Peek shouldn't change size
    queue.peek(1)
    assert queue.size() == 2
    
    # Pop should reduce size
    queue.pop(1)
    assert queue.size() == 1
    
    print("✅ Peek vs pop behavior correct")

if __name__ == "__main__":
    test_queue_initialization()
    test_push_and_pop()
    test_priority_ordering()
    test_recency_boost()
    test_max_size_enforcement()
    test_stale_event_removal()
    test_peek_vs_pop()
    
    print("\n" + "="*50)
    print("🎉 ALL PRIORITY QUEUE TESTS PASSED!")
    print("="*50)