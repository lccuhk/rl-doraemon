import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import Trainer

def main():
    print("="*60)
    print("测试详细日志测试")
    print("="*60)
    
    env = MahjongEnv(seed=42)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong",
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
        seed=42
    )
    
    opponent = RandomAgent("Opponent", seed=43)
    
    agents = [dqn_agent, opponent, opponent, opponent]
    
    trainer = Trainer(env, agents, log_dir="./logs", checkpoint_dir="./checkpoints")
    
    num_episodes = 10
    print(f"\n开始训练 {num_episodes} 回合...")
    print(f"设备: {dqn_agent.device}")
    
    history = trainer.train(
        num_episodes=num_episodes,
        save_freq=100,
        eval_freq=100,
        num_eval_episodes=5
    )
    
    print("\n训练完成！")
    print(f"总步数: {dqn_agent.steps_done}")
    print(f"最终epsilon: {dqn_agent.get_epsilon():.4f}")

if __name__ == "__main__":
    main()
