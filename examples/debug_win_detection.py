import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv, ActionType, GamePhase
from src.environment.rules import MahjongRules
from src.agents.random_agent import RandomAgent

def main():
    print("="*70)
    print("调试和牌检测逻辑")
    print("="*70)
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    opponent = RandomAgent("Opponent", seed=43)
    agents = [opponent, opponent, opponent, opponent]
    
    num_episodes = 50
    tsumo_checks = 0
    ron_checks = 0
    hand_size_14 = 0
    hand_size_13 = 0
    win_detected = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0
        
        for agent in agents:
            agent.reset()
        
        while not done:
            step += 1
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            phase = obs['phase']
            
            player = env.players[current_player]
            hand_size = len(player['hand'])
            
            if phase == GamePhase.DRAWING.value:
                tsumo_checks += 1
                if hand_size == 14:
                    hand_size_14 += 1
                    
                    is_win, win_info = MahjongRules.check_tsumo(
                        player['hand'],
                        env.melds[current_player],
                        env.is_menzen[current_player]
                    )
                    
                    if is_win:
                        win_detected += 1
                        print(f"\n[回合{episode+1} 步骤{step}] 检测到自摸!")
                        print(f"  手牌数量: {hand_size}")
                        print(f"  合法动作: {legal_actions}")
                        print(f"  TSUMO在合法动作中: {ActionType.TSUMO.value in legal_actions}")
                        print(f"  和牌信息: {win_info}")
                elif hand_size == 13:
                    hand_size_13 += 1
            
            if phase == GamePhase.RESPONDING.value and env.last_discard is not None:
                ron_checks += 1
                
                is_win, win_info = MahjongRules.check_ron(
                    player['hand'],
                    env.last_discard,
                    env.melds[current_player],
                    env.is_menzen[current_player]
                )
                
                if is_win:
                    win_detected += 1
                    print(f"\n[回合{episode+1} 步骤{step}] 检测到荣和!")
                    print(f"  手牌数量: {hand_size}")
                    print(f"  合法动作: {legal_actions}")
                    print(f"  RON在合法动作中: {ActionType.RON.value in legal_actions}")
                    print(f"  和牌信息: {win_info}")
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            obs, reward, done, info = env.step(action)
        
        if env.winner >= 0:
            print(f"\n[回合{episode+1}] 玩家{env.winner} 和牌成功!")
    
    print("\n" + "="*70)
    print("统计结果")
    print("="*70)
    print(f"总回合数: {num_episodes}")
    print(f"自摸检查次数: {tsumo_checks}")
    print(f"  - 手牌14张的次数: {hand_size_14}")
    print(f"  - 手牌13张的次数: {hand_size_13}")
    print(f"荣和检查次数: {ron_checks}")
    print(f"检测到和牌次数: {win_detected}")
    print("="*70)

if __name__ == "__main__":
    main()
