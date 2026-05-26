import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase, ActionType
from src.agents.random_agent import RandomAgent
from src.utils.tile_utils import TileUtils

def debug_game_win():
    print("="*60)
    print("调试游戏中的和牌检测")
    print("="*60)
    
    env = MahjongEnv(seed=42)
    
    print("\n--- 初始状态 ---")
    obs = env.reset()
    print(f"玩家数: {len(env.players)}")
    print(f"牌山剩余: {len(env.wall)}")
    
    for i in range(4):
        hand = env.players[i]['hand']
        symbols = [TileUtils.get_tile_symbol(t) for t in hand]
        print(f"玩家{i}手牌 ({len(hand)}张): {' '.join(symbols)}")
    
    print("\n--- 模拟游戏流程 ---")
    done = False
    step_count = 0
    max_steps = 200
    
    win_opportunities = []
    
    while not done and step_count < max_steps:
        current_player = obs['current_player']
        legal_actions = obs['legal_actions']
        phase = GamePhase(obs['phase'])
        
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
            print(f"    合法动作: {legal_actions}")
            
            hand = obs['hand']
            symbols = [TileUtils.get_tile_symbol(t) for t in hand]
            print(f"    玩家{current_player}手牌: {' '.join(symbols)}")
        
        action = legal_actions[0] if legal_actions else 0
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if 'win_type' in info:
            print(f"\n🎉 和牌成功! 玩家{current_player}, {info['win_type']}")
            if 'win_info' in info:
                win_info = info['win_info']
                print(f"   番数: {win_info.get('han', 0)}, 符数: {win_info.get('fu', 0)}")
                print(f"   役种: {win_info.get('yaku', [])}")
    
    print(f"\n--- 游戏结束 ---")
    print(f"总步数: {step_count}")
    print(f"和牌机会次数: {len(win_opportunities)}")
    print(f"获胜者: {env.winner}")
    
    if env.winner >= 0:
        print(f"获胜者类型: {'玩家0' if env.winner == 0 else '其他玩家'}")
    else:
        print("流局")
    
    print(f"\n最终分数:")
    for i, player in enumerate(env.players):
        print(f"  玩家{i}: {player['score']}点")

if __name__ == "__main__":
    debug_game_win()
