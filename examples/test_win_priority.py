import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env_debug import MahjongEnvDebug
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent_debug import DQNAgentDebug

def main():
    print("="*70)
    print("测试优先和牌逻辑")
    print("="*70)
    
    print("\n【测试说明】")
    print("  - 运行10个回合，使用带详细日志的调试版本")
    print("  - 当有和牌机会时，90%概率应该优先选择和牌动作")
    print("  - 日志会显示：[和牌机会]、[随机-优先和牌]、[放弃自摸-随机]等")
    print("="*70 + "\n")
    
    env = MahjongEnvDebug(seed=42, use_intermediate_rewards=True, debug_level=1)
    
    dqn_agent = DQNAgentDebug(
        name="DQN_Test",
        input_shape=(6, 34, 4),
        num_actions=41,
        hidden_size=256,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=1000,
        buffer_capacity=100000,
        seed=42,
        debug_level=1
    )
    
    opponent = RandomAgent("Opponent", seed=43)
    agents = [dqn_agent, opponent, opponent, opponent]
    
    num_episodes = 10
    print(f"\n开始测试 {num_episodes} 回合...\n")
    
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
            print(f"  结果: 玩家{env.winner} 和牌！")
        else:
            print(f"  结果: 流局")
    
    print("\n" + "="*70)
    print("测试完成！统计摘要")
    print("="*70)
    
    dqn_agent.print_action_stats()
    
    env_stats = env.get_debug_stats()
    print(f"\n环境统计:")
    print(f"  总回合数: {env_stats['episode_count']}")
    print(f"  实际和牌次数: {env_stats['win_detected_count']}")
    print(f"  自摸机会: {env_stats['win_opportunities']['tsumo_available']}")
    print(f"  自摸成功: {env_stats['win_opportunities']['tsumo_taken']}")
    print(f"  荣和机会: {env_stats['win_opportunities']['ron_available']}")
    print(f"  荣和成功: {env_stats['win_opportunities']['ron_taken']}")
    print("="*70)

if __name__ == "__main__":
    main()
