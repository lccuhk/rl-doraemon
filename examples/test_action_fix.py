import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv, ActionType, GamePhase
from src.agents.random_agent import RandomAgent

def main():
    print("="*70)
    print("测试动作编号修复")
    print("="*70)
    
    print(f"\nActionType 定义:")
    print(f"  CHI = {ActionType.CHI.value}")
    print(f"  PON = {ActionType.PON.value}")
    print(f"  KAN = {ActionType.KAN.value}")
    print(f"  RIICHI = {ActionType.RIICHI.value}")
    print(f"  RON = {ActionType.RON.value}")
    print(f"  TSUMO = {ActionType.TSUMO.value}")
    print(f"  PASS = {ActionType.PASS.value}")
    print(f"\n舍牌动作范围: 0-33")
    print(f"特殊动作范围: 34-40")
    print(f"无冲突: {ActionType.CHI.value > 33}")
    
    print("\n" + "="*70)
    print("运行100回合测试，统计和牌机会")
    print("="*70)
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    opponent = RandomAgent("Opponent", seed=43)
    agents = [opponent, opponent, opponent, opponent]
    
    num_episodes = 100
    tsumo_opportunities = 0
    ron_opportunities = 0
    actual_wins = 0
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        
        for agent in agents:
            agent.reset()
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            phase = obs['phase']
            
            if ActionType.TSUMO.value in legal_actions:
                tsumo_opportunities += 1
                print(f"[回合{episode+1}] 玩家{current_player} 有自摸机会! 合法动作: {legal_actions}")
            
            if ActionType.RON.value in legal_actions:
                ron_opportunities += 1
                print(f"[回合{episode+1}] 玩家{current_player} 有荣和机会! 合法动作: {legal_actions}")
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            obs, reward, done, info = env.step(action)
        
        if env.winner >= 0:
            actual_wins += 1
            print(f"[回合{episode+1}] 玩家{env.winner} 和牌成功!")
    
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    print(f"总回合数: {num_episodes}")
    print(f"自摸机会: {tsumo_opportunities}")
    print(f"荣和机会: {ron_opportunities}")
    print(f"实际和牌: {actual_wins}")
    print(f"和牌率: {actual_wins/num_episodes*100:.1f}%")
    print("="*70)

if __name__ == "__main__":
    main()
