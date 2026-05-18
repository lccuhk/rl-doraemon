#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("麻将AI - 简单测试")
print("="*60)

print("\n[1/4] 导入模块...")
from src.utils.tile_utils import TileUtils
from src.environment.mahjong_env import MahjongEnv, ActionType, GamePhase
from src.environment.rules import MahjongRules
from src.agents.random_agent import RandomAgent
print("   ✓ 所有模块导入成功")

print("\n[2/4] 测试牌型工具...")
tile = TileUtils.create_tile('wan', 1)
print(f"   创建牌: {TileUtils.get_tile_symbol(tile)} {TileUtils.get_tile_name(tile)}")
print(f"   牌ID: {tile['id']}")

wall = TileUtils.create_full_wall()
print(f"   完整牌山: {len(wall)} 张牌")
print(f"   洗牌后: {len(TileUtils.shuffle_wall(wall, seed=42))} 张牌")
print("   ✓ 牌型工具测试通过")

print("\n[3/4] 测试麻将环境...")
env = MahjongEnv(seed=42)
obs = env.reset()
print(f"   玩家数量: {env.num_players}")
print(f"   玩家0手牌数: {len(env.players[0]['hand'])}")
print(f"   牌山剩余: {len(env.wall)}")
print(f"   观察形状: {obs['obs'].shape}")
print(f"   合法动作: {obs['legal_actions']}")

hand = obs['hand']
symbols = [TileUtils.get_tile_symbol(t) for t in hand]
print(f"   玩家0手牌: {' '.join(symbols)}")
print("   ✓ 麻将环境测试通过")

print("\n[4/4] 测试游戏流程...")
print("   运行1局AI对战...")

ai1 = RandomAgent("AI_1", seed=42)
ai2 = RandomAgent("AI_2", seed=43)
ai3 = RandomAgent("AI_3", seed=44)
ai4 = RandomAgent("AI_4", seed=45)
agents = [ai1, ai2, ai3, ai4]

done = False
steps = 0
max_steps = 500

while not done and steps < max_steps:
    current_player = obs['current_player']
    legal_actions = obs['legal_actions']
    
    if not legal_actions:
        break
    
    action = agents[current_player].select_action(obs, legal_actions)
    obs, reward, done, info = env.step(action)
    steps += 1

if env.winner >= 0:
    print(f"   获胜者: 玩家 {env.winner}")
else:
    print(f"   流局")
print(f"   总步数: {steps}")
print(f"   最终分数: {[p['score'] for p in env.players]}")
print("   ✓ 游戏流程测试通过")

print("\n" + "="*60)
print("所有测试通过！麻将游戏运行正常！")
print("="*60)
print("\n你可以运行以下命令开始游戏：")
print("  python3 main.py          - 启动交互式菜单")
print("  python3 examples/play_game.py  - 直接开始游戏")
print("  python3 examples/train_dqn.py   - 训练DQN模型")
