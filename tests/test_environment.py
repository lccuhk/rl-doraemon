import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import numpy as np
from src.environment.mahjong_env import MahjongEnv, ActionType, GamePhase
from src.environment.rules import MahjongRules
from src.utils.tile_utils import TileUtils

class TestTileUtils(unittest.TestCase):
    def test_tile_creation(self):
        tile = TileUtils.create_tile('wan', 1)
        self.assertEqual(tile['type'], 'wan')
        self.assertEqual(tile['value'], 1)
        self.assertEqual(tile['id'], 0)
    
    def test_tile_id_conversion(self):
        tile_type, value = TileUtils.id_to_tile(0)
        self.assertEqual(tile_type, 'wan')
        self.assertEqual(value, 1)
        
        tile_id = TileUtils.tile_to_id('tong', 9)
        self.assertEqual(tile_id, 26)
    
    def test_full_wall_creation(self):
        wall = TileUtils.create_full_wall()
        self.assertEqual(len(wall), 136)
        
        counts = TileUtils.count_tiles(wall)
        for tid in range(34):
            self.assertEqual(counts[tid], 4)
    
    def test_shuffle_wall(self):
        wall = TileUtils.create_full_wall()
        shuffled = TileUtils.shuffle_wall(wall, seed=42)
        self.assertEqual(len(shuffled), 136)
        self.assertNotEqual(wall[0]['id'], shuffled[0]['id'])

class TestMahjongRules(unittest.TestCase):
    def test_check_pon(self):
        hand = [
            TileUtils.create_tile('wan', 1),
            TileUtils.create_tile('wan', 1),
            TileUtils.create_tile('wan', 2),
        ]
        tile = TileUtils.create_tile('wan', 1)
        self.assertTrue(MahjongRules.check_pon(hand, tile))
        
        tile2 = TileUtils.create_tile('wan', 3)
        self.assertFalse(MahjongRules.check_pon(hand, tile2))
    
    def test_check_chi(self):
        hand = [
            TileUtils.create_tile('wan', 2),
            TileUtils.create_tile('wan', 3),
            TileUtils.create_tile('wan', 5),
        ]
        tile = TileUtils.create_tile('wan', 1)
        chis = MahjongRules.check_chi(hand, tile)
        self.assertEqual(len(chis), 1)
    
    def test_check_chiitoitsu(self):
        hand = []
        for i in range(1, 8):
            hand.extend([TileUtils.create_tile('wan', i)] * 2)
        
        is_chiitoitsu, info = MahjongRules.check_chiitoitsu(hand)
        self.assertTrue(is_chiitoitsu)
    
    def test_score_calculation(self):
        score_info = MahjongRules.calculate_score(
            han=3, fu=40, is_tsumo=False, is_dealer=False
        )
        self.assertGreater(score_info['total_win'], 0)

class TestMahjongEnv(unittest.TestCase):
    def setUp(self):
        self.env = MahjongEnv(seed=42)
    
    def test_env_initialization(self):
        obs = self.env.reset()
        self.assertEqual(self.env.num_players, 4)
        self.assertEqual(len(self.env.players), 4)
        self.assertEqual(len(self.env.players[0]['hand']), 13)
        self.assertEqual(len(self.env.wall), 70)
    
    def test_observation_shape(self):
        obs = self.env.reset()
        self.assertEqual(obs['obs'].shape, (6, 34, 4))
    
    def test_legal_actions(self):
        obs = self.env.reset()
        legal_actions = obs['legal_actions']
        self.assertIsInstance(legal_actions, list)
        self.assertGreater(len(legal_actions), 0)
    
    def test_step_function(self):
        obs = self.env.reset()
        
        while self.env.phase != GamePhase.DISCARDING:
            legal_actions = obs['legal_actions']
            if ActionType.RIICHI.value in legal_actions:
                action = legal_actions[0] if legal_actions[0] != ActionType.RIICHI.value else 0
            else:
                action = legal_actions[0] if legal_actions else 0
            obs, reward, done, info = self.env.step(action)
            if done:
                break
        
        if not self.env.game_over:
            legal_actions = obs['legal_actions']
            if legal_actions:
                action = legal_actions[0]
                obs, reward, done, info = self.env.step(action)
                
                self.assertIn('obs', obs)
                self.assertIsInstance(reward, float)
                self.assertIsInstance(done, bool)
                self.assertIsInstance(info, dict)
    
    def test_game_flow(self):
        obs = self.env.reset()
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            legal_actions = obs['legal_actions']
            if legal_actions:
                action = legal_actions[0]
            else:
                break
            
            obs, reward, done, info = self.env.step(action)
            steps += 1
        
        self.assertLessEqual(steps, max_steps)

if __name__ == '__main__':
    unittest.main(verbosity=2)
