from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class BaseAgent(ABC):
    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.player_id: int = -1
    
    @abstractmethod
    def select_action(self, observation: Dict, legal_actions: List[int]) -> int:
        pass
    
    def observe(self, observation: Dict, action: int, reward: float, 
                next_observation: Dict, done: bool):
        pass
    
    def reset(self):
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
