import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv, ActionType, GamePhase
from src.environment.rules import MahjongRules
from src.utils.tile_utils import TileUtils
from src.agents.random_agent import RandomAgent

def main():
    print("="*70)
    print("调试游戏流程")
    print("="*70)
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    opponent = RandomAgent("Opponent", seed=43)
    agents = [opponent, opponent, opponent, opponent]
    
    num_episodes = 10
    
    for episode in range(num_episodes):
        print(f"\n--- 第 {episode + 1} 回合 ---")
        
        obs = env.reset()
        done = False
        step = 0
        
        for agent in agents:
            agent.reset()
        
        while not done and step < 50:
            step += 1
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            phase = obs['phase']
            
            player = env.players[current_player]
            hand_size = len(player['hand'])
            
            if phase == GamePhase.DRAWING.value:
                print(f"[步骤{step}] 玩家{current_player} DRAWING阶段, 手牌:{hand_size}张")
                
                if hand_size == 14:
                    is_win, win_info = MahjongRules.check_tsumo(
                        player['hand'],
                        env.melds[current_player],
                        env.is_menzen[current_player]
                    )
                    
                    if is_win:
                        print(f"  -> 检测到自摸! 合法动作包含TSUMO({ActionType.TSUMO.value}): {ActionType.TSUMO.value in legal_actions}")
                        print(f"  -> 合法动作: {legal_actions}")
                        print(f"  -> 和牌信息: {win_info}")
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            obs, reward, done, info = env.step(action)
        
        if env.winner >= 0:
            print(f"  -> 玩家{env.winner} 和牌成功!")
        else:
            print(f"  -> 流局 (步数:{step})")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
