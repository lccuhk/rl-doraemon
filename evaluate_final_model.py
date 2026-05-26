import sys
import os
import glob
import argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
import numpy as np
from tqdm import tqdm
from datetime import datetime

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
    
    snapshots = sorted(glob.glob(os.path.join(checkpoint_dir, 'snapshots', '*_agent0.pt')))
    
    if snapshots:
        return snapshots[-1], "snapshot"
    
    return None, None

def test_agent(agent, num_episodes=100, seed=42):
    env = MahjongEnv(seed=seed)
    opponent = RandomAgent("Opponent", seed=seed + 1)
    
    agents = [agent, opponent, opponent, opponent]
    
    stats = {
        'episodes': 0,
        'wins': 0,
        'losses': 0,
        'draws': 0,
        'tsumo_wins': 0,
        'ron_wins': 0,
        'total_rewards': [],
        'episode_lengths': [],
        'win_types': [],
        'yaku_types': []
    }
    
    for episode in tqdm(range(num_episodes), desc=f"测试 {agent.name}"):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        
        for a in agents:
            a.reset()
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            
            action = agents[current_player].select_action(obs, legal_actions)
            next_obs, reward, done, info = env.step(action)
            
            if current_player == 0:
                episode_reward += reward
            
            obs = next_obs
            episode_steps += 1
        
        stats['episodes'] += 1
        stats['total_rewards'].append(episode_reward)
        stats['episode_lengths'].append(episode_steps)
        
        if env.winner == 0:
            stats['wins'] += 1
            if 'win_type' in info:
                stats['win_types'].append(info['win_type'])
                if info['win_type'] == 'tsumo':
                    stats['tsumo_wins'] += 1
                elif info['win_type'] == 'ron':
                    stats['ron_wins'] += 1
            if 'yaku' in info:
                stats['yaku_types'].extend(info['yaku'])
        elif env.winner >= 0:
            stats['losses'] += 1
        else:
            stats['draws'] += 1
    
    return stats

def print_stats(stats, agent_name, model_path=None):
    print("\n" + "="*70)
    print(f"{agent_name} 评估测试结果")
    print("="*70)
    
    if model_path:
        print(f"模型文件: {model_path}")
        print("-"*70)
    
    print(f"总回合数: {stats['episodes']}")
    print(f"\n胜负统计:")
    print(f"  胜利次数: {stats['wins']} ({stats['wins']/stats['episodes']*100:.1f}%)")
    print(f"  失败次数: {stats['losses']} ({stats['losses']/stats['episodes']*100:.1f}%)")
    print(f"  流局次数: {stats['draws']} ({stats['draws']/stats['episodes']*100:.1f}%)")
    
    print(f"\n和牌类型:")
    print(f"  自摸胜利: {stats['tsumo_wins']}")
    print(f"  荣和胜利: {stats['ron_wins']}")
    
    if stats['total_rewards']:
        print(f"\n奖励统计:")
        print(f"  平均奖励: {np.mean(stats['total_rewards']):.4f}")
        print(f"  最小奖励: {np.min(stats['total_rewards']):.4f}")
        print(f"  最大奖励: {np.max(stats['total_rewards']):.4f}")
        print(f"  奖励标准差: {np.std(stats['total_rewards']):.4f}")
        print(f"  奖励中位数: {np.median(stats['total_rewards']):.4f}")
    
    if stats['episode_lengths']:
        print(f"\n回合长度统计:")
        print(f"  平均步数: {np.mean(stats['episode_lengths']):.1f}")
        print(f"  最小步数: {np.min(stats['episode_lengths'])}")
        print(f"  最大步数: {np.max(stats['episode_lengths'])}")
    
    if stats['yaku_types']:
        print(f"\n役种统计:")
        yaku_counts = {}
        for yaku in stats['yaku_types']:
            yaku_counts[yaku] = yaku_counts.get(yaku, 0) + 1
        for yaku, count in sorted(yaku_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {yaku}: {count} 次")
    
    print("="*70)

def compare_with_random(trained_stats, random_stats):
    print("\n" + "="*70)
    print("模型对比分析")
    print("="*70)
    
    trained_win_rate = trained_stats['wins'] / trained_stats['episodes'] * 100
    random_win_rate = random_stats['wins'] / random_stats['episodes'] * 100
    
    trained_avg_reward = np.mean(trained_stats['total_rewards'])
    random_avg_reward = np.mean(random_stats['total_rewards'])
    
    print(f"\n胜率对比:")
    print(f"  DQN 模型: {trained_win_rate:.1f}%")
    print(f"  随机策略: {random_win_rate:.1f}%")
    print(f"  提升: {trained_win_rate - random_win_rate:+.1f}%")
    
    print(f"\n平均奖励对比:")
    print(f"  DQN 模型: {trained_avg_reward:.4f}")
    print(f"  随机策略: {random_avg_reward:.4f}")
    print(f"  提升: {trained_avg_reward - random_avg_reward:+.4f}")
    
    print(f"\n和牌次数对比:")
    print(f"  DQN 模型: {trained_stats['tsumo_wins'] + trained_stats['ron_wins']} 次")
    print(f"  随机策略: {random_stats['tsumo_wins'] + random_stats['ron_wins']} 次")
    
    if trained_win_rate > random_win_rate:
        print(f"\n✅ DQN 模型表现优于随机策略！")
    elif trained_win_rate < random_win_rate:
        print(f"\n⚠️  DQN 模型表现不如随机策略，需要更多训练")
    else:
        print(f"\n⚖️  DQN 模型与随机策略表现相当")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='评估训练好的麻将AI模型')
    parser.add_argument('--model', type=str, default=None, help='指定模型文件路径')
    parser.add_argument('--episodes', type=int, default=100, help='测试回合数（默认100）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--compare', action='store_true', help='与随机策略对比')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints_2000', help='检查点目录')
    args = parser.parse_args()
    
    print("="*70)
    print("麻将AI模型评估测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试回合数: {args.episodes}")
    print(f"随机种子: {args.seed}")
    print("="*70)
    
    model_path = args.model
    model_type = "specified"
    
    if not model_path:
        print("\n正在查找最新模型...")
        model_path, model_type = find_latest_model(args.checkpoint_dir)
        
        if not model_path:
            print(f"\n错误: 在 {args.checkpoint_dir} 中没有找到模型文件")
            print("请先完成训练或指定模型文件路径")
            return
    
    print(f"\n找到模型: {model_path}")
    print(f"模型类型: {model_type}")
    
    print("\n" + "="*70)
    print("加载模型...")
    print("="*70)
    
    trained_agent = DQNAgent(
        name="DQN_Trained",
        input_shape=(6, 34, 4),
        num_actions=41,
        hidden_size=256,
        epsilon_start=0.01,
        epsilon_end=0.01
    )
    
    trained_agent.load(model_path)
    trained_agent.set_training(False)
    print(f"模型加载成功！")
    print(f"  训练步数: {trained_agent.steps_done}")
    print(f"  探索率: {trained_agent.get_epsilon():.4f}")
    
    print(f"\n开始测试 DQN 模型 ({args.episodes} 回合)...")
    trained_stats = test_agent(trained_agent, num_episodes=args.episodes, seed=args.seed)
    
    print_stats(trained_stats, "DQN 训练模型", model_path)
    
    if args.compare:
        print(f"\n开始测试随机策略 ({args.episodes} 回合)...")
        random_agent = RandomAgent("Random", seed=args.seed + 100)
        random_stats = test_agent(random_agent, num_episodes=args.episodes, seed=args.seed + 100)
        
        print_stats(random_stats, "随机策略")
        compare_with_random(trained_stats, random_stats)
    
    print("\n" + "="*70)
    print("评估测试完成！")
    print("="*70)

if __name__ == "__main__":
    main()
