#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase
from src.agents.random_agent import RandomAgent
from src.utils.tile_utils import TileUtils

def auto_ai_demo():
    print("="*60)
    print("🀄 麻将AI - AI对战演示")
    print("="*60)
    
    print("\n初始化游戏环境...")
    env = MahjongEnv(seed=42)
    
    ai1 = RandomAgent("AI_东", seed=42)
    ai2 = RandomAgent("AI_南", seed=43)
    ai3 = RandomAgent("AI_西", seed=44)
    ai4 = RandomAgent("AI_北", seed=45)
    agents = [ai1, ai2, ai3, ai4]
    
    obs = env.reset()
    done = False
    step_count = 0
    max_steps = 200
    
    print("\n" + "="*60)
    print("游戏开始！")
    print("="*60)
    
    while not done and step_count < max_steps:
        current_player = obs['current_player']
        legal_actions = obs['legal_actions']
        
        if step_count < 10 or step_count % 20 == 0:
            print(f"\n--- 第 {step_count + 1} 步 ---")
            print(f"当前玩家: 玩家{current_player} ({agents[current_player].name})")
            print(f"当前阶段: {GamePhase(obs['phase']).name}")
            
            if current_player == 0:
                hand = obs['hand']
                symbols = [TileUtils.get_tile_symbol(t) for t in hand]
                print(f"玩家0手牌: {' '.join(symbols)}")
            
            print(f"合法动作: {legal_actions}")
        
        if not legal_actions:
            print("警告: 没有合法动作！")
            break
        
        action = agents[current_player].select_action(obs, legal_actions)
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if 'discarded' in info:
            tile = info['discarded']
            if step_count < 10 or step_count % 20 == 0:
                print(f"玩家{current_player} 舍牌: {TileUtils.get_tile_symbol(tile)} {TileUtils.get_tile_name(tile)}")
        
        if 'win_type' in info:
            print(f"\n🎉 玩家{current_player} {info['win_type']}！")
            if 'win_info' in info:
                win_info = info['win_info']
                print(f"   番数: {win_info.get('han', 0)}")
                print(f"   符数: {win_info.get('fu', 0)}")
                if 'yaku' in win_info:
                    print(f"   役种: {win_info['yaku']}")
    
    print("\n" + "="*60)
    print("游戏结束！")
    print("="*60)
    
    if env.winner >= 0:
        print(f"\n🏆 获胜者: 玩家{env.winner} ({agents[env.winner].name})")
    else:
        print("\n🤝 流局")
    
    print(f"\n总步数: {step_count}")
    print("\n最终分数:")
    for i, player in enumerate(env.players):
        status = []
        if i == env.dealer:
            status.append("庄家")
        if env.is_riichi[i]:
            status.append("立直")
        status_str = f" ({', '.join(status)})" if status else ""
        print(f"  玩家{i} ({agents[i].name}): {player['score']}点{status_str}")
    
    print("\n舍牌记录:")
    for i in range(4):
        if env.discards[i]:
            symbols = [TileUtils.get_tile_symbol(t) for t in env.discards[i]]
            print(f"  玩家{i}: {' '.join(symbols)}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)

if __name__ == "__main__":
    auto_ai_demo()
