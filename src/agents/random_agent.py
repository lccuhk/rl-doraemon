from typing import Dict, List, Optional
import numpy as np
from .base_agent import BaseAgent

class RandomAgent(BaseAgent):
    def __init__(self, name: str = "RandomAgent", seed: Optional[int] = None):
        super().__init__(name)
        self.rng = np.random.RandomState(seed)
    
    def select_action(self, observation: Dict, legal_actions: List[int]) -> int:
        if not legal_actions:
            return 0
        
        return self.rng.choice(legal_actions)
    
    def reset(self):
        pass
