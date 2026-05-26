import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
import numpy as np
from tqdm import tqdm

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
        'win_types': []
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
        elif env.winner >= 0:
            stats['losses'] += 1
        else:
            stats['draws'] += 1
    
    return stats

def print_stats(stats, agent_name):
    print("\n" + "="*60)
    print(f"{agent_name} 测试结果")
    print("="*60)
    print(f"总回合数: {stats['episodes']}")
    print(f"胜利次数: {stats['wins']} ({stats['wins']/stats['episodes']*100:.1f}%)")
    print(f"失败次数: {stats['losses']} ({stats['losses']/stats['episodes']*100:.1f}%)")
    print(f"流局次数: {stats['draws']} ({stats['draws']/stats['episodes']*100:.1f}%)")
    print(f"自摸胜利: {stats['tsumo_wins']}")
    print(f"荣和胜利: {stats['ron_wins']}")
    
    if stats['total_rewards']:
        print(f"\n奖励统计:")
        print(f"  平均奖励: {np.mean(stats['total_rewards']):.4f}")
        print(f"  最小奖励: {np.min(stats['total_rewards']):.4f}")
        print(f"  最大奖励: {np.max(stats['total_rewards']):.4f}")
        print(f"  奖励标准差: {np.std(stats['total_rewards']):.4f}")
    
    if stats['episode_lengths']:
        print(f"\n回合长度统计:")
        print(f"  平均步数: {np.mean(stats['episode_lengths']):.1f}")
        print(f"  最小步数: {np.min(stats['episode_lengths'])}")
        print(f"  最大步数: {np.max(stats['episode_lengths'])}")
    
    if stats['win_types']:
        print(f"\n和牌类型统计:")
        win_type_counts = {}
        for wt in stats['win_types']:
            win_type_counts[wt] = win_type_counts.get(wt, 0) + 1
        for wt, count in win_type_counts.items():
            print(f"  {wt}: {count} 次")
    
    print("="*60)

def main():
    print("="*60)
    print("模型对比测试")
    print("="*60)
    
    num_episodes = 100
    print(f"\n测试回合数: {num_episodes}")
    
    print("\n" + "="*60)
    print("加载训练好的模型...")
    print("="*60)
    
    trained_agent = DQNAgent(
        name="DQN_Trained",
        input_shape=(6, 34, 4),
        num_actions=41,
        hidden_size=256
    )
    
    model_path = "./checkpoints/trained_model.pt"
    if os.path.exists(model_path):
        trained_agent.load(model_path)
        trained_agent.training = False
        print(f"模型加载成功！步数: {trained_agent.steps_done}, epsilon: {trained_agent.get_epsilon():.4f}")
    else:
        print(f"警告: 模型文件不存在: {model_path}")
        print("使用随机初始化的模型进行测试")
    
    random_agent = RandomAgent("Random", seed=123)
    
    print(f"\n开始测试训练好的模型 ({num_episodes} 回合)...")
    trained_stats = test_agent(trained_agent, num_episodes=num_episodes, seed=42)
    
    print(f"\n开始测试随机策略 ({num_episodes} 回合)...")
    random_stats = test_agent(random_agent, num_episodes=num_episodes, seed=42)
    
    print_stats(trained_stats, "训练好的 DQN 模型")
    print_stats(random_stats, "随机策略")
    
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)
    
    trained_win_rate = trained_stats['wins'] / trained_stats['episodes'] * 100
    random_win_rate = random_stats['wins'] / random_stats['episodes'] * 100
    
    print(f"\n胜率对比:")
    print(f"  训练模型胜率: {trained_win_rate:.1f}%")
    print(f"  随机策略胜率: {random_win_rate:.1f}%")
    print(f"  胜率提升: {trained_win_rate - random_win_rate:+.1f}%")
    
    trained_avg_reward = np.mean(trained_stats['total_rewards'])
    random_avg_reward = np.mean(random_stats['total_rewards'])
    
    print(f"\n平均奖励对比:")
    print(f"  训练模型平均奖励: {trained_avg_reward:.4f}")
    print(f"  随机策略平均奖励: {random_avg_reward:.4f}")
    print(f"  奖励提升: {trained_avg_reward - random_avg_reward:+.4f}")
    
    if trained_win_rate > random_win_rate:
        print(f"\n✅ 训练模型表现优于随机策略！")
    elif trained_win_rate < random_win_rate:
        print(f"\n⚠️  训练模型表现不如随机策略，需要更多训练")
    else:
        print(f"\n⚖️  训练模型与随机策略表现相当")
    
    print("="*60)

if __name__ == "__main__":
    main()
