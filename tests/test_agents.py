import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent, ReplayBuffer
from src.environment.mahjong_env import MahjongEnv

class TestRandomAgent(unittest.TestCase):
    def setUp(self):
        self.agent = RandomAgent(seed=42)
        self.env = MahjongEnv(seed=42)
    
    def test_select_action(self):
        obs = self.env.reset()
        legal_actions = obs['legal_actions']
        
        action = self.agent.select_action(obs, legal_actions)
        self.assertIn(action, legal_actions)
    
    def test_reset(self):
        self.agent.reset()

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_push(self):
        state = np.random.rand(6, 34, 4)
        self.buffer.push(state, 0, 1.0, state, False, [0, 1, 2])
        self.assertEqual(len(self.buffer), 1)
    
    def test_sample(self):
        for _ in range(10):
            state = np.random.rand(6, 34, 4)
            self.buffer.push(state, 0, 1.0, state, False, [0, 1, 2])
        
        batch = self.buffer.sample(5)
        self.assertEqual(len(batch[0]), 5)
    
    def test_capacity(self):
        for _ in range(150):
            state = np.random.rand(6, 34, 4)
            self.buffer.push(state, 0, 1.0, state, False, [0, 1, 2])
        
        self.assertEqual(len(self.buffer), 100)

class TestDQNAgent(unittest.TestCase):
    def setUp(self):
        self.agent = DQNAgent(
            input_shape=(6, 34, 4),
            num_actions=38,
            hidden_size=64,
            batch_size=8,
            seed=42
        )
        self.env = MahjongEnv(seed=42)
    
    def test_initialization(self):
        self.assertIsNotNone(self.agent.policy_net)
        self.assertIsNotNone(self.agent.target_net)
        self.assertEqual(self.agent.epsilon, 1.0)
    
    def test_select_action_training(self):
        obs = self.env.reset()
        legal_actions = obs['legal_actions']
        
        self.agent.set_training(True)
        action = self.agent.select_action(obs, legal_actions)
        self.assertIsInstance(action, int)
    
    def test_select_action_eval(self):
        obs = self.env.reset()
        legal_actions = obs['legal_actions']
        
        self.agent.set_training(False)
        action = self.agent.select_action(obs, legal_actions)
        self.assertIsInstance(action, int)
    
    def test_observe(self):
        obs = self.env.reset()
        next_obs = obs.copy()
        
        for _ in range(10):
            self.agent.observe(obs, 0, 1.0, next_obs, False)
        
        self.assertGreater(len(self.agent.replay_buffer), 0)
    
    def test_save_load(self):
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model.pt')
            self.agent.save(path)
            self.assertTrue(os.path.exists(path))
            
            new_agent = DQNAgent(
                input_shape=(6, 34, 4),
                num_actions=38,
                hidden_size=64
            )
            new_agent.load(path)

if __name__ == '__main__':
    unittest.main(verbosity=2)
