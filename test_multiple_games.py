import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase, ActionType
from src.agents.random_agent import RandomAgent
from src.utils.tile_utils import TileUtils

def test_multiple_games(num_games=10):
    print("="*70)
    print(f"测试 {num_games} 局游戏")
    print("="*70)
    
    total_wins = 0
    total_draws = 0
    total_win_opportunities = 0
    
    for game_num in range(num_games):
        print(f"\n--- 第 {game_num + 1} 局 ---")
        
        env = MahjongEnv(seed=game_num)
        obs = env.reset()
        
        done = False
        step_count = 0
        max_steps = 200
        
        win_opportunities = 0
        game_won = False
        
        while not done and step_count < max_steps:
            current_player = env.current_player
            phase = env.phase
            
            legal_actions = env._get_legal_actions(current_player)
            
            has_tsumo = ActionType.TSUMO.value in legal_actions
            has_ron = ActionType.RON.value in legal_actions
            
            if has_tsumo or has_ron:
                win_opportunities += 1
                total_win_opportunities += 1
                win_type = "自摸" if has_tsumo else "荣和"
                print(f"  ⚠️  和牌机会! 第{step_count}步, 玩家{current_player}, {win_type}")
            
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
                total_wins += 1
                game_won = True
                print(f"  🎉 和牌成功! 玩家{current_player}, {info['win_type']}")
                if 'win_info' in info:
                    win_info = info['win_info']
                    print(f"     番数: {win_info.get('han', 0)}, 符数: {win_info.get('fu', 0)}")
                    print(f"     役种: {win_info.get('yaku', [])}")
                break
        
        if not game_won and done:
            total_draws += 1
            print(f"  🤝 流局")
        
        print(f"  总步数: {step_count}, 和牌机会: {win_opportunities}")
    
    print(f"\n{'='*70}")
    print("统计结果")
    print(f"{'='*70}")
    print(f"总局数: {num_games}")
    print(f"和牌成功: {total_wins}")
    print(f"流局: {total_draws}")
    print(f"总共和牌机会: {total_win_opportunities}")
    print(f"和牌率: {total_wins/num_games*100:.1f}%")
    print(f"每局平均和牌机会: {total_win_opportunities/num_games:.1f}")
    
    return total_wins > 0

if __name__ == "__main__":
    success = test_multiple_games(10)
    if success:
        print("\n✅ 修复成功！有和牌成功的局。")
    else:
        print("\n⚠️  游戏流程正常，但这10局都没有和牌成功。")
