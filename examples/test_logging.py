import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent

def main():
    print("="*70)
    print("测试详细日志功能")
    print("="*70)
    
    print("\n【日志说明】")
    print("  - [和牌机会]: 检测到自摸或荣和机会")
    print("  - [优先和牌-随机]: 随机阶段优先选择和牌动作")
    print("  - [放弃和牌-探索]: 随机阶段为了探索放弃和牌")
    print("  - [贪心决策]: 贪心选择时的Q值对比")
    print("  - [自摸和牌!]/[荣和和牌!]: 实际和牌时的详细信息")
    print("="*70 + "\n")
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    
    dqn_agent = DQNAgent(
        name="DQN_Test",
        input_shape=(6, 34, 4),
        num_actions=41,
        hidden_size=256,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.5,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=1000,
        buffer_capacity=100000,
        seed=42
    )
    
    opponent = RandomAgent("Opponent", seed=43)
    agents = [dqn_agent, opponent, opponent, opponent]
    
    num_episodes = 20
    print(f"运行 {num_episodes} 回合测试...\n")
    
    for episode in range(num_episodes):
        print(f"\n--- 第 {episode + 1} 回合 ---")
        
        obs = env.reset()
        done = False
        
        for agent in agents:
            agent.reset()
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            obs, reward, done, info = env.step(action)
        
        if env.winner >= 0:
            print(f"  -> 玩家{env.winner} 和牌成功!")
        else:
            print(f"  -> 流局")
    
    print("\n" + "="*70)
    print("测试完成!")
    print("="*70)

if __name__ == "__main__":
    main()
