import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import Trainer

def main():
    print("="*60)
    print("麻将AI - DQN训练示例 (带中间奖励)")
    print("="*60)
    
    print("\n中间奖励系统:")
    print("  - 向听数改进: +0.25/每向听")
    print("  - 听牌(向听数=0): +0.5")
    print("  - 立直: +0.3")
    print("  - 杠: +0.2")
    print("  - 碰: +0.15")
    print("  - 吃: +0.1")
    print("  - 打孤张字牌: +0.05")
    print("  - 打孤张幺九: +0.03")
    print("  - 向听数恶化: -0.1")
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong_v2",
        input_shape=(6, 34, 4),
        num_actions=38,
        hidden_size=256,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        batch_size=64,
        target_update_freq=1000,
        buffer_capacity=100000,
        seed=42
    )
    
    opponent = RandomAgent("Opponent", seed=43)
    
    agents = [dqn_agent, opponent, opponent, opponent]
    
    trainer = Trainer(env, agents, log_dir="./logs_v2", checkpoint_dir="./checkpoints_v2")
    
    num_episodes = 500
    print(f"\n开始训练 {num_episodes} 回合...")
    print(f"设备: {dqn_agent.device}")
    
    history = trainer.train(
        num_episodes=num_episodes,
        save_freq=50,
        eval_freq=50,
        num_eval_episodes=10
    )
    
    print("\n训练完成！")
    print(f"总步数: {dqn_agent.steps_done}")
    print(f"最终epsilon: {dqn_agent.get_epsilon():.4f}")
    
    print("\n评估最终模型...")
    dqn_agent.set_training(False)
    results = trainer.tournament(num_games=50)
    
    print("\n" + "="*60)
    print("最终评估结果")
    print("="*60)
    print(f"DQN Agent 胜率: {results['win_rates'][0]*100:.1f}%")
    print(f"DQN Agent 平均分数: {results['avg_scores'][0]:.0f}")
    print(f"对手平均胜率: {sum(results['win_rates'][1:])/3*100:.1f}%")
    
    print("\n详细统计:")
    for i in range(4):
        agent_type = "DQN" if i == 0 else "Random"
        print(f"  玩家{i} ({agent_type}): 胜率={results['win_rates'][i]*100:.1f}%, 平均分={results['avg_scores'][i]:.0f}")

if __name__ == "__main__":
    main()
