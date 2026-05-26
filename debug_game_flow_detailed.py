import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase, ActionType
from src.agents.random_agent import RandomAgent
from src.utils.tile_utils import TileUtils

def debug_game_flow_detailed():
    print("="*70)
    print("详细调试游戏流程")
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
    max_steps = 30
    
    while not done and step_count < max_steps:
        current_player = env.current_player
        phase = env.phase
        
        hand = env.players[current_player]['hand']
        hand_symbols = [TileUtils.get_tile_symbol(t) for t in hand]
        
        print(f"\n第{step_count}步: 玩家{current_player}, 阶段: {phase.name}")
        print(f"  手牌数: {len(hand)}")
        print(f"  牌山剩余: {len(env.wall)}")
        
        legal_actions = env._get_legal_actions(current_player)
        print(f"  合法动作: {legal_actions}")
        
        if phase == GamePhase.DRAWING:
            if ActionType.TSUMO.value in legal_actions:
                print(f"  ⚠️  可以自摸!")
            if ActionType.KAN.value in legal_actions:
                print(f"  ⚠️  可以暗杠!")
            if ActionType.RIICHI.value in legal_actions:
                print(f"  ⚠️  可以立直!")
            
            action = ActionType.PASS.value if ActionType.PASS.value in legal_actions else 0
            print(f"  选择动作: PASS (进入舍牌阶段)")
        
        elif phase == GamePhase.DISCARDING:
            action = 0
            print(f"  选择动作: 舍牌 (动作 0)")
        
        elif phase == GamePhase.RESPONDING:
            if ActionType.RON.value in legal_actions:
                print(f"  ⚠️  可以荣和!")
            if ActionType.PON.value in legal_actions:
                print(f"  ⚠️  可以碰!")
            if ActionType.CHI.value in legal_actions:
                print(f"  ⚠️  可以吃!")
            
            action = ActionType.PASS.value if ActionType.PASS.value in legal_actions else 0
            print(f"  选择动作: PASS")
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if 'win_type' in info:
            print(f"\n🎉 和牌成功! {info['win_type']}")
            break
        
        if done:
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

if __name__ == "__main__":
    debug_game_flow_detailed()
