#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.training.trainer import Trainer
from src.utils.tile_utils import TileUtils

def test_game_startup():
    print("="*60)
    print("测试麻将游戏启动")
    print("="*60)
    
    print("\n1. 创建麻将环境...")
    env = MahjongEnv(seed=42)
    
    print("2. 重置游戏...")
    obs = env.reset()
    
    print(f"   - 玩家数量: {env.num_players}")
    print(f"   - 每个玩家手牌数: {len(env.players[0]['hand'])}")
    print(f"   - 牌山剩余: {len(env.wall)}")
    print(f"   - 观察形状: {obs['obs'].shape}")
    
    print("\n3. 显示玩家0的手牌:")
    hand = obs['hand']
    symbols = [TileUtils.get_tile_symbol(t) for t in hand]
    names = [TileUtils.get_tile_name(t) for t in hand]
    print(f"   符号: {' '.join(symbols)}")
    print(f"   名称: {' '.join(names)}")
    
    print("\n4. 宝牌指示牌:")
    for tile in obs['dora_indicators']:
        print(f"   {TileUtils.get_tile_symbol(tile)} {TileUtils.get_tile_name(tile)}")
    
    print("\n" + "="*60)
    print("AI对战测试")
    print("="*60)
    
    ai1 = RandomAgent("AI_1", seed=42)
    ai2 = RandomAgent("AI_2", seed=43)
    ai3 = RandomAgent("AI_3", seed=44)
    ai4 = RandomAgent("AI_4", seed=45)
    
    agents = [ai1, ai2, ai3, ai4]
    trainer = Trainer(env, agents)
    
    print("\n运行1局AI对战...")
    winner, scores = trainer.play_game(render=False)
    
    if winner >= 0:
        print(f"   获胜者: 玩家 {winner} ({agents[winner].name})")
    else:
        print("   流局")
    print(f"   最终分数: {scores}")
    
    print("\n" + "="*60)
    print("锦标赛测试 (10局)")
    print("="*60)
    
    results = trainer.tournament(num_games=10)
    
    print(f"\n结果:")
    print(f"   总局数: {results['total_games']}")
    print(f"   流局数: {results['draws']}")
    print("\n   各玩家胜率:")
    for i, rate in enumerate(results['win_rates']):
        print(f"      玩家{i} ({agents[i].name}: {rate*100:.1f}%")
    print("\n   平均分数:")
    for i, score in enumerate(results['avg_scores']):
        print(f"      玩家{i}: {score:.0f}")
    
    print("\n" + "="*60)
    print("测试完成！游戏运行正常！")
    print("="*60)

if __name__ == "__main__":
    test_game_startup()
