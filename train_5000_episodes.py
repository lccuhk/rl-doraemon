import sys
import os
import glob
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import Trainer

def find_latest_model(checkpoint_dir="./checkpoints_2000"):
    final_models = sorted(glob.glob(os.path.join(checkpoint_dir, 'final_model_*_agent0.pt')))
    if final_models:
        return final_models[-1], "final"
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*_agent0.pt'))
    if checkpoint_files:
        def get_episode_number(filepath):
            filename = os.path.basename(filepath)
            try:
                return int(filename.split('_')[1])
            except:
                return -1
        
        checkpoint_files.sort(key=get_episode_number)
        return checkpoint_files[-1], "checkpoint"
    
    trained_models = glob.glob(os.path.join(checkpoint_dir, 'trained_model_*.pt'))
    if trained_models:
        return trained_models[-1], "trained"
    
    return None, None

def main():
    parser = argparse.ArgumentParser(description='麻将AI DQN训练 - 5000回合')
    parser.add_argument('--resume', action='store_true', help='从最新模型恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None, help='指定检查点路径')
    parser.add_argument('--start-episode', type=int, default=2000, help='起始回合号')
    parser.add_argument('--episodes', type=int, default=5000, help='总回合数')
    parser.add_argument('--save-freq', type=int, default=50, help='检查点保存频率（回合）')
    parser.add_argument('--snapshot-freq', type=int, default=50, help='快照保存频率（回合，0表示禁用）')
    parser.add_argument('--max-snapshots', type=int, default=10, help='保留的快照数量')
    args = parser.parse_args()
    
    print("="*70)
    print("麻将AI - DQN训练 (5000回合)")
    print("="*70)
    
    env = MahjongEnv(seed=42)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong_5000",
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
    
    start_episode = args.start_episode
    checkpoint_path = args.checkpoint
    
    if args.resume and not checkpoint_path:
        checkpoint_path, model_type = find_latest_model()
        if checkpoint_path:
            print(f"\n发现模型: {checkpoint_path}")
            print(f"模型类型: {model_type}")
            print(f"将从第 {start_episode} 回合继续训练")
        else:
            print("\n未找到模型，将从头开始训练")
            start_episode = 0
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n加载模型: {checkpoint_path}")
        dqn_agent.load(checkpoint_path)
        print(f"模型加载成功！")
        print(f"  已训练步数: {dqn_agent.steps_done}")
        print(f"  当前探索率: {dqn_agent.get_epsilon():.4f}")
    
    opponent = RandomAgent("Opponent", seed=43)
    
    agents = [dqn_agent, opponent, opponent, opponent]
    
    trainer = Trainer(env, agents, log_dir="./logs_5000", checkpoint_dir="./checkpoints_5000")
    
    num_episodes = args.episodes
    print(f"\n开始训练 {num_episodes} 回合...")
    print(f"起始回合: {start_episode + 1}")
    print(f"设备: {dqn_agent.device}")
    print(f"检查点保存频率: 每 {args.save_freq} 回合")
    if args.snapshot_freq > 0:
        print(f"快照保存频率: 每 {args.snapshot_freq} 回合")
        print(f"保留最近快照数: {args.max_snapshots}")
    else:
        print(f"快照保存: 禁用")
    
    print("\n" + "="*70)
    print("信号控制说明:")
    print("  - 手动保存快照: kill -SIGUSR1 <PID>")
    print("  - 优雅退出: kill -SIGTERM <PID> 或 Ctrl+C")
    print("="*70)
    
    history = trainer.train(
        num_episodes=num_episodes,
        save_freq=args.save_freq,
        eval_freq=100,
        num_eval_episodes=10,
        snapshot_freq=args.snapshot_freq,
        max_snapshots=args.max_snapshots,
        start_episode=start_episode
    )
    
    print("\n训练完成！")
    print(f"总步数: {dqn_agent.steps_done}")
    print(f"最终epsilon: {dqn_agent.get_epsilon():.4f}")
    
    os.makedirs("./checkpoints_5000", exist_ok=True)
    model_path = "./checkpoints_5000/trained_model_5000.pt"
    dqn_agent.save(model_path)
    print(f"\n模型已保存到: {model_path}")

if __name__ == "__main__":
    main()
