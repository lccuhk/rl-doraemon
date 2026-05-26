import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase, ActionType
from src.agents.random_agent import RandomAgent
from src.utils.tile_utils import TileUtils

def debug_game_flow():
    print("="*60)
    print("详细调试游戏流程")
    print("="*60)
    
    env = MahjongEnv(seed=42)
    
    print("\n--- 初始状态 ---")
    obs = env.reset()
    
    for i in range(4):
        hand = env.players[i]['hand']
        symbols = [TileUtils.get_tile_symbol(t) for t in hand]
        print(f"玩家{i}手牌 ({len(hand)}张): {' '.join(symbols)}")
    
    print(f"\n牌山剩余: {len(env.wall)}")
    
    print("\n--- 开始游戏 ---")
    done = False
    step_count = 0
    max_steps = 50
    
    win_opportunities = []
    
    while not done and step_count < max_steps:
        current_player = obs['current_player']
        legal_actions = obs['legal_actions']
        phase = GamePhase(obs['phase'])
        
        hand = obs['hand']
        hand_symbols = [TileUtils.get_tile_symbol(t) for t in hand]
        
        has_tsumo = ActionType.TSUMO.value in legal_actions
        has_ron = ActionType.RON.value in legal_actions
        
        print(f"\n第{step_count}步: 玩家{current_player}, 阶段: {phase.name}")
        print(f"  手牌数: {len(hand)}")
        print(f"  手牌: {' '.join(hand_symbols)}")
        print(f"  合法动作: {legal_actions}")
        
        if has_tsumo:
            print(f"  ⚠️  可以自摸!")
            win_opportunities.append({
                'step': step_count,
                'player': current_player,
                'type': 'tsumo',
                'hand': hand_symbols
            })
        
        if has_ron:
            print(f"  ⚠️  可以荣和!")
            win_opportunities.append({
                'step': step_count,
                'player': current_player,
                'type': 'ron',
                'hand': hand_symbols
            })
        
        if phase == GamePhase.DRAWING:
            if ActionType.TSUMO.value in legal_actions:
                print(f"  选择动作: 自摸 (TSUMO)")
                action = ActionType.TSUMO.value
            elif ActionType.PASS.value in legal_actions:
                print(f"  选择动作: PASS")
                action = ActionType.PASS.value
            else:
                action = legal_actions[0] if legal_actions else 0
        elif phase == GamePhase.DISCARDING:
            print(f"  选择动作: 舍牌 (动作 {legal_actions[0] if legal_actions else 0})")
            action = legal_actions[0] if legal_actions else 0
        else:
            if ActionType.PASS.value in legal_actions:
                print(f"  选择动作: PASS")
                action = ActionType.PASS.value
            else:
                action = legal_actions[0] if legal_actions else 0
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if 'win_type' in info:
            print(f"\n🎉 和牌成功! {info['win_type']}")
            if 'win_info' in info:
                win_info = info['win_info']
                print(f"   番数: {win_info.get('han', 0)}, 符数: {win_info.get('fu', 0)}")
                print(f"   役种: {win_info.get('yaku', [])}")
            break
    
    print(f"\n--- 游戏结束 ---")
    print(f"总步数: {step_count}")
    print(f"和牌机会次数: {len(win_opportunities)}")
    print(f"获胜者: {env.winner}")
    
    if win_opportunities:
        print(f"\n和牌机会详情:")
        for opp in win_opportunities:
            print(f"  第{opp['step']}步: 玩家{opp['player']} - {opp['type']}")
            print(f"    手牌: {' '.join(opp['hand'])}")

if __name__ == "__main__":
    debug_game_flow()
