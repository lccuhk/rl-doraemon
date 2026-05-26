import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase, ActionType
from src.agents.random_agent import RandomAgent
from src.utils.tile_utils import TileUtils

def test_fixed_game_flow():
    print("="*70)
    print("测试修复后的游戏流程")
    print("="*70)
    
    env = MahjongEnv(seed=42)
    
    print("\n--- 初始状态 ---")
    obs = env.reset()
    
    for i in range(4):
        hand = env.players[i]['hand']
        print(f"玩家{i}手牌数: {len(hand)}")
    
    print(f"\n牌山剩余: {len(env.wall)}")
    print(f"当前玩家: {env.current_player}")
    print(f"当前阶段: {env.phase.name}")
    
    print("\n--- 开始游戏 ---")
    done = False
    step_count = 0
    max_steps = 200
    
    win_opportunities = []
    wins = 0
    draws = 0
    
    while not done and step_count < max_steps:
        current_player = env.current_player
        phase = env.phase
        
        hand = env.players[current_player]['hand']
        
        legal_actions = env._get_legal_actions(current_player)
        
        has_tsumo = ActionType.TSUMO.value in legal_actions
        has_ron = ActionType.RON.value in legal_actions
        
        if has_tsumo or has_ron:
            win_type = "自摸" if has_tsumo else "荣和"
            win_opportunities.append({
                'step': step_count,
                'player': current_player,
                'type': win_type,
                'phase': phase.name
            })
            print(f"\n⚠️  和牌机会! 第{step_count}步, 玩家{current_player}, {win_type}, 阶段: {phase.name}")
            print(f"    手牌数: {len(hand)}")
        
        if phase == GamePhase.DRAWING:
            if ActionType.TSUMO.value in legal_actions:
                action = ActionType.TSUMO.value
            elif ActionType.PASS.value in legal_actions:
                action = ActionType.PASS.value
            else:
                action = 0
        elif phase == GamePhase.DISCARDING:
            action = 0
        else:
            if ActionType.RON.value in legal_actions:
                action = ActionType.RON.value
            elif ActionType.PON.value in legal_actions:
                action = ActionType.PON.value
            elif ActionType.CHI.value in legal_actions:
                action = ActionType.CHI.value
            elif ActionType.PASS.value in legal_actions:
                action = ActionType.PASS.value
            else:
                action = 0
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if 'win_type' in info:
            wins += 1
            print(f"\n🎉 和牌成功! 玩家{current_player}, {info['win_type']}")
            if 'win_info' in info:
                win_info = info['win_info']
                print(f"   番数: {win_info.get('han', 0)}, 符数: {win_info.get('fu', 0)}")
                print(f"   役种: {win_info.get('yaku', [])}")
            break
        
        if done:
            draws += 1
            print(f"\n--- 游戏结束 ---")
            print(f"获胜者: {env.winner}")
            if env.winner < 0:
                print("流局")
            break
    
    print(f"\n--- 最终状态 ---")
    for i in range(4):
        hand = env.players[i]['hand']
        print(f"玩家{i}手牌数: {len(hand)}")
    print(f"牌山剩余: {len(env.wall)}")
    
    print(f"\n--- 统计 ---")
    print(f"总步数: {step_count}")
    print(f"和牌机会次数: {len(win_opportunities)}")
    print(f"和牌成功: {wins}")
    print(f"流局: {draws}")
    
    if win_opportunities:
        print(f"\n和牌机会详情:")
        for opp in win_opportunities:
            print(f"  第{opp['step']}步: 玩家{opp['player']} - {opp['type']}")
    
    return len(win_opportunities) > 0

if __name__ == "__main__":
    success = test_fixed_game_flow()
    if success:
        print("\n✅ 修复成功！游戏流程正常，有和牌机会。")
    else:
        print("\n⚠️  游戏流程正常，但这局没有和牌机会。")
