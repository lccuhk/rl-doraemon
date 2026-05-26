import sys
import os
import glob
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import Trainer

def find_latest_checkpoint(checkpoint_dir="./checkpoints_2000"):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*_agent0.pt'))
    if not checkpoint_files:
        return None, 0
    
    def get_episode_number(filepath):
        filename = os.path.basename(filepath)
        try:
            return int(filename.split('_')[1])
        except:
            return -1
    
    checkpoint_files.sort(key=get_episode_number)
    latest_file = checkpoint_files[-1]
    episode = get_episode_number(latest_file)
    
    if episode >= 0:
        return latest_file, episode + 1
    else:
        return latest_file, 0

def main():
    parser = argparse.ArgumentParser(description='麻将AI DQN训练')
    parser.add_argument('--resume', action='store_true', help='从最新检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None, help='指定检查点路径')
    parser.add_argument('--start-episode', type=int, default=0, help='起始回合号')
    parser.add_argument('--episodes', type=int, default=2000, help='总回合数')
    parser.add_argument('--save-freq', type=int, default=50, help='检查点保存频率（回合）')
    parser.add_argument('--snapshot-freq', type=int, default=50, help='快照保存频率（回合，0表示禁用）')
    parser.add_argument('--max-snapshots', type=int, default=10, help='保留的快照数量')
    args = parser.parse_args()
    
    print("="*60)
    print("麻将AI - DQN训练 (2000回合)")
    print("="*60)
    
    env = MahjongEnv(seed=42)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong_2000",
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
        checkpoint_path, start_episode = find_latest_checkpoint()
        if checkpoint_path:
            print(f"\n发现检查点: {checkpoint_path}")
            print(f"将从第 {start_episode} 回合恢复训练")
        else:
            print("\n未找到检查点，将从头开始训练")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\n加载检查点: {checkpoint_path}")
        dqn_agent.load(checkpoint_path)
        print("检查点加载成功！")
    
    opponent = RandomAgent("Opponent", seed=43)
    
    agents = [dqn_agent, opponent, opponent, opponent]
    
    trainer = Trainer(env, agents, log_dir="./logs_2000", checkpoint_dir="./checkpoints_2000")
    
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
    
    print("\n" + "="*60)
    print("信号控制说明:")
    print("  - 手动保存快照: kill -SIGUSR1 <PID>")
    print("  - 优雅退出: kill -SIGTERM <PID> 或 Ctrl+C")
    print("="*60)
    
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
    
    os.makedirs("./checkpoints_2000", exist_ok=True)
    model_path = "./checkpoints_2000/trained_model_2000.pt"
    dqn_agent.save(model_path)
    print(f"\n模型已保存到: {model_path}")

if __name__ == "__main__":
    main()
