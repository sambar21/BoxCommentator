from typing import List, Optional

class StateNode:
    def __init__(self, name: str, priority: float, contexts: List[str]):
        self.name = name
        self.priority = priority
        self.contexts = contexts
        self.next: Optional['StateNode'] = None
        self.prev: Optional['StateNode'] = None