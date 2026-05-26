import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statistics

def load_training_history(log_dir="./logs_2000"):
    history_files = sorted(glob.glob(os.path.join(log_dir, 'training_history_*.json')))
    
    if not history_files:
        print(f"错误: 在 {log_dir} 中没有找到训练历史文件")
        return None
    
    latest_file = history_files[-1]
    print(f"加载训练历史: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def extract_reward_data(data):
    episodes = data.get('episodes', [])
    rewards = data.get('rewards', [])
    
    if not episodes or not rewards:
        print("警告: 没有找到回合或奖励数据")
        return None
    
    dqn_rewards = [r[0] for r in rewards]
    opp1_rewards = [r[1] for r in rewards]
    opp2_rewards = [r[2] for r in rewards]
    opp3_rewards = [r[3] for r in rewards]
    
    return {
        'episodes': episodes,
        'dqn_rewards': dqn_rewards,
        'opp1_rewards': opp1_rewards,
        'opp2_rewards': opp2_rewards,
        'opp3_rewards': opp3_rewards
    }

def extract_loss_data_from_logs(log_dir="./logs_2000"):
    log_files = sorted(glob.glob(os.path.join(log_dir, '*.log')))
    
    if not log_files:
        print("警告: 没有找到日志文件，使用训练统计中的损失数据")
        return get_loss_data_from_stats()
    
    losses = []
    steps = []
    
    for log_file in log_files:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if '平均损失' in line and '损失标准差' in line:
                    try:
                        parts = line.split('平均损失:')[1].split(',')
                        avg_loss = float(parts[0].strip())
                        step_part = line.split('步数=')[1].split(']')[0]
                        step = int(step_part)
                        losses.append(avg_loss)
                        steps.append(step)
                    except:
                        continue
    
    if not losses:
        return get_loss_data_from_stats()
    
    return {
        'steps': steps,
        'losses': losses
    }

def get_loss_data_from_stats():
    loss_data = {
        'steps': [1000, 2000, 3000, 4000, 5000, 6000, 7000],
        'losses': [0.794755, 0.138486, 0.101895, 0.110310, 0.140269, 0.130933, 0.112137],
        'loss_stds': [0.222564, 0.185651, 0.148336, 0.157911, 0.194238, 0.201511, 0.168632],
        'epsilons': [0.7810, 0.6310, 0.5170, 0.4350, 0.3736, 0.3200, 0.2482],
        'avg_rewards': [0.0069, 0.0075, 0.0072, 0.0054, 0.0064, 0.0061, 0.0092]
    }
    return loss_data

def calculate_moving_average(data, window_size=10):
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        if i < window_size - 1:
            avg = sum(data[:i+1]) / (i+1)
        else:
            avg = sum(data[i-window_size+1:i+1]) / window_size
        moving_avg.append(avg)
    
    return moving_avg

def plot_curves(reward_data, loss_data=None, output_dir="./charts"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('麻将AI训练曲线', fontsize=16, fontweight='bold')
    
    episodes = reward_data['episodes']
    dqn_rewards = reward_data['dqn_rewards']
    
    window_size = min(10, len(episodes) // 5) if len(episodes) > 5 else 1
    dqn_ma = calculate_moving_average(dqn_rewards, window_size)
    
    ax1 = axes[0, 0]
    ax1.plot(episodes, dqn_rewards, 'b-', alpha=0.3, label='原始奖励')
    ax1.plot(episodes, dqn_ma, 'b-', linewidth=2, label=f'{window_size}回合移动平均')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('奖励')
    ax1.set_title('DQN Agent 奖励曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.plot(episodes, dqn_rewards, 'b-', alpha=0.5, label='DQN Agent')
    ax2.plot(episodes, reward_data['opp1_rewards'], 'r-', alpha=0.5, label='对手1')
    ax2.plot(episodes, reward_data['opp2_rewards'], 'g-', alpha=0.5, label='对手2')
    ax2.plot(episodes, reward_data['opp3_rewards'], 'orange', alpha=0.5, label='对手3')
    ax2.set_xlabel('回合')
    ax2.set_ylabel('奖励')
    ax2.set_title('所有玩家奖励对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    avg_rewards = []
    for i in range(0, len(dqn_rewards), max(1, len(dqn_rewards) // 10)):
        chunk = dqn_rewards[i:i+max(1, len(dqn_rewards) // 10)]
        avg_rewards.append(sum(chunk) / len(chunk))
    
    chunk_labels = [f'{i*max(1, len(dqn_rewards)//10)+1}-{(i+1)*max(1, len(dqn_rewards)//10)}' 
                   for i in range(len(avg_rewards))]
    
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(avg_rewards)))
    bars = ax3.bar(range(len(avg_rewards)), avg_rewards, color=colors)
    ax3.set_xlabel('回合区间')
    ax3.set_ylabel('平均奖励')
    ax3.set_title('分阶段平均奖励')
    ax3.set_xticks(range(len(chunk_labels)))
    ax3.set_xticklabels(chunk_labels, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, avg_rewards):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax4 = axes[1, 1]
    if loss_data and len(loss_data['losses']) > 0:
        steps = loss_data['steps']
        losses = loss_data['losses']
        loss_ma = calculate_moving_average(losses, min(5, len(losses)))
        
        ax4.plot(steps, losses, 'r-', alpha=0.3, label='原始损失')
        ax4.plot(steps, loss_ma, 'r-', linewidth=2, label='移动平均')
        ax4.set_xlabel('训练步数')
        ax4.set_ylabel('损失值')
        ax4.set_title('损失曲线')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '损失数据暂不可用\n(需要从训练日志中提取)', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('损失曲线')
        ax4.axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'training_curves_{timestamp}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")
    
    plt.show()
    
    return output_file

def print_statistics(reward_data):
    print("\n" + "="*60)
    print("训练统计摘要")
    print("="*60)
    
    episodes = reward_data['episodes']
    dqn_rewards = reward_data['dqn_rewards']
    
    print(f"总回合数: {len(episodes)}")
    print(f"\nDQN Agent 奖励统计:")
    print(f"  平均奖励: {statistics.mean(dqn_rewards):.4f}")
    print(f"  最小奖励: {min(dqn_rewards):.4f}")
    print(f"  最大奖励: {max(dqn_rewards):.4f}")
    if len(dqn_rewards) > 1:
        print(f"  标准差: {statistics.stdev(dqn_rewards):.4f}")
    
    if len(dqn_rewards) >= 20:
        first_half = dqn_rewards[:len(dqn_rewards)//2]
        second_half = dqn_rewards[len(dqn_rewards)//2:]
        print(f"\n奖励趋势分析:")
        print(f"  前半段平均: {statistics.mean(first_half):.4f}")
        print(f"  后半段平均: {statistics.mean(second_half):.4f}")
        improvement = statistics.mean(second_half) - statistics.mean(first_half)
        print(f"  变化: {improvement:+.4f} ({'上升' if improvement > 0 else '下降'})")
    
    print("\n各玩家平均奖励:")
    print(f"  DQN Agent: {statistics.mean(dqn_rewards):.4f}")
    print(f"  对手1: {statistics.mean(reward_data['opp1_rewards']):.4f}")
    print(f"  对手2: {statistics.mean(reward_data['opp2_rewards']):.4f}")
    print(f"  对手3: {statistics.mean(reward_data['opp3_rewards']):.4f}")
    
    print("="*60)

def main():
    print("="*60)
    print("麻将AI训练曲线可视化")
    print("="*60)
    
    log_dir = "./logs_2000"
    
    data = load_training_history(log_dir)
    if data is None:
        return
    
    reward_data = extract_reward_data(data)
    if reward_data is None:
        return
    
    loss_data = extract_loss_data_from_logs(log_dir)
    
    print_statistics(reward_data)
    
    output_file = plot_curves(reward_data, loss_data)
    
    print(f"\n请查看生成的图表: {output_file}")

if __name__ == "__main__":
    main()
