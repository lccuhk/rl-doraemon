import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import statistics

def load_convergence_log(log_dir="./logs_5000"):
    log_files = sorted(glob.glob(os.path.join(log_dir, 'convergence_log_*.json')))
    
    if not log_files:
        print(f"错误: 在 {log_dir} 中没有找到收敛日志文件")
        return None
    
    latest_file = log_files[-1]
    print(f"加载收敛日志: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def load_training_history(log_dir="./logs_5000"):
    history_files = sorted(glob.glob(os.path.join(log_dir, 'training_history_*.json')))
    
    if not history_files:
        print(f"警告: 在 {log_dir} 中没有找到训练历史文件")
        return None
    
    latest_file = history_files[-1]
    print(f"加载训练历史: {latest_file}")
    
    with open(latest_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def extract_convergence_data(convergence_data):
    if not convergence_data or 'training_stats' not in convergence_data:
        return None
    
    training_stats = convergence_data['training_stats']
    
    steps = []
    losses = []
    rewards = []
    epsilons = []
    loss_stds = []
    reward_stds = []
    
    for entry in training_stats:
        stats = entry.get('stats', {})
        
        step = stats.get('steps', 0)
        if step < 1000:
            continue
        
        steps.append(step)
        losses.append(stats.get('avg_loss', 0))
        rewards.append(stats.get('avg_reward', 0))
        epsilons.append(stats.get('epsilon', 0))
        loss_stds.append(stats.get('loss_std', 0))
        reward_stds.append(stats.get('reward_std', 0))
    
    return {
        'steps': steps,
        'losses': losses,
        'rewards': rewards,
        'epsilons': epsilons,
        'loss_stds': loss_stds,
        'reward_stds': reward_stds
    }

def extract_reward_data(history_data):
    if not history_data:
        return None
    
    episodes = history_data.get('episodes', [])
    rewards = history_data.get('rewards', [])
    
    if not episodes or not rewards:
        return None
    
    dqn_rewards = [r[0] for r in rewards]
    
    return {
        'episodes': episodes,
        'dqn_rewards': dqn_rewards
    }

def calculate_moving_average(data, window_size=5):
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

def plot_convergence_curves(convergence_data, reward_data=None, output_dir="./charts"):
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('麻将AI训练收敛曲线', fontsize=16, fontweight='bold')
    
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    
    if convergence_data:
        steps = convergence_data['steps']
        losses = convergence_data['losses']
        rewards = convergence_data['rewards']
        epsilons = convergence_data['epsilons']
        loss_stds = convergence_data['loss_stds']
        reward_stds = convergence_data['reward_stds']
        
        loss_ma = calculate_moving_average(losses, window_size=3)
        reward_ma = calculate_moving_average(rewards, window_size=3)
        
        ax1.plot(steps, losses, 'b-', alpha=0.3, label='原始损失')
        ax1.plot(steps, loss_ma, 'b-', linewidth=2, label='移动平均')
        ax1.fill_between(steps, 
                        np.array(losses) - np.array(loss_stds),
                        np.array(losses) + np.array(loss_stds),
                        alpha=0.2, color='blue', label='标准差范围')
        ax1.set_xlabel('训练步数')
        ax1.set_ylabel('损失值')
        ax1.set_title('损失曲线')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(steps, rewards, 'g-', alpha=0.3, label='原始奖励')
        ax2.plot(steps, reward_ma, 'g-', linewidth=2, label='移动平均')
        ax2.fill_between(steps,
                        np.array(rewards) - np.array(reward_stds),
                        np.array(rewards) + np.array(reward_stds),
                        alpha=0.2, color='green', label='标准差范围')
        ax2.set_xlabel('训练步数')
        ax2.set_ylabel('平均奖励')
        ax2.set_title('奖励曲线')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(steps, epsilons, 'r-', linewidth=2)
        ax3.set_xlabel('训练步数')
        ax3.set_ylabel('探索率 (%)')
        ax3.set_title('探索率衰减曲线')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='最小值 1%')
        ax3.legend()
        
        ax4.plot(steps, loss_stds, 'purple', linewidth=2, label='损失标准差')
        ax4.plot(steps, reward_stds, 'orange', linewidth=2, label='奖励标准差')
        ax4.set_xlabel('训练步数')
        ax4.set_ylabel('标准差')
        ax4.set_title('稳定性分析（标准差）')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    if reward_data:
        episodes = reward_data['episodes']
        dqn_rewards = reward_data['dqn_rewards']
        
        window_size = min(20, len(episodes) // 5) if len(episodes) > 5 else 1
        reward_ma = calculate_moving_average(dqn_rewards, window_size=window_size)
        
        ax5.scatter(episodes, dqn_rewards, alpha=0.3, s=10, label='单回合奖励')
        ax5.plot(episodes, reward_ma, 'r-', linewidth=2, label=f'{window_size}回合移动平均')
        ax5.set_xlabel('回合数')
        ax5.set_ylabel('DQN Agent 奖励')
        ax5.set_title('回合奖励分布（最近训练）')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        if len(dqn_rewards) >= 10:
            z = np.polyfit(episodes, dqn_rewards, 1)
            p = np.poly1d(z)
            ax5.plot(episodes, p(episodes), 'k--', alpha=0.5, 
                    label=f'趋势线 (斜率={z[0]:.4f})')
            ax5.legend()
    else:
        ax5.text(0.5, 0.5, '回合奖励数据暂不可用', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('回合奖励分布')
        ax5.axis('off')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'convergence_curves_{timestamp}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")
    
    plt.show()
    
    return output_file

def print_statistics(convergence_data, reward_data=None):
    print("\n" + "="*70)
    print("收敛统计摘要")
    print("="*70)
    
    if convergence_data:
        steps = convergence_data['steps']
        losses = convergence_data['losses']
        rewards = convergence_data['rewards']
        epsilons = convergence_data['epsilons']
        
        print(f"\n训练统计数据点: {len(steps)}")
        
        if len(losses) > 0:
            print(f"\n损失统计:")
            print(f"  初始损失: {losses[0]:.6f}")
            print(f"  最终损失: {losses[-1]:.6f}")
            print(f"  最小损失: {min(losses):.6f}")
            print(f"  最大损失: {max(losses):.6f}")
            print(f"  平均损失: {statistics.mean(losses):.6f}")
            if len(losses) > 1:
                print(f"  损失变化: {losses[-1] - losses[0]:+.6f}")
        
        if len(rewards) > 0:
            print(f"\n奖励统计:")
            print(f"  初始奖励: {rewards[0]:.4f}")
            print(f"  最终奖励: {rewards[-1]:.4f}")
            print(f"  最小奖励: {min(rewards):.4f}")
            print(f"  最大奖励: {max(rewards):.4f}")
            print(f"  平均奖励: {statistics.mean(rewards):.4f}")
            if len(rewards) > 1:
                print(f"  奖励变化: {rewards[-1] - rewards[0]:+.4f}")
        
        if len(epsilons) > 0:
            print(f"\n探索率统计:")
            print(f"  初始探索率: {epsilons[0]:.2f}%")
            print(f"  最终探索率: {epsilons[-1]:.2f}%")
    
    if reward_data:
        dqn_rewards = reward_data['dqn_rewards']
        episodes = reward_data['episodes']
        
        print(f"\n回合奖励统计 (共 {len(episodes)} 回合):")
        print(f"  平均奖励: {statistics.mean(dqn_rewards):.4f}")
        print(f"  最小奖励: {min(dqn_rewards):.4f}")
        print(f"  最大奖励: {max(dqn_rewards):.4f}")
        if len(dqn_rewards) > 1:
            print(f"  标准差: {statistics.stdev(dqn_rewards):.4f}")
        
        positive_rewards = sum(1 for r in dqn_rewards if r > 0)
        print(f"  正奖励回合: {positive_rewards} ({positive_rewards/len(dqn_rewards)*100:.1f}%)")
        
        if len(dqn_rewards) >= 20:
            first_half = dqn_rewards[:len(dqn_rewards)//2]
            second_half = dqn_rewards[len(dqn_rewards)//2:]
            print(f"\n奖励趋势分析:")
            print(f"  前半段平均: {statistics.mean(first_half):.4f}")
            print(f"  后半段平均: {statistics.mean(second_half):.4f}")
            improvement = statistics.mean(second_half) - statistics.mean(first_half)
            print(f"  变化: {improvement:+.4f} ({'上升' if improvement > 0 else '下降'})")
    
    print("="*70)

def main():
    print("="*70)
    print("麻将AI训练收敛曲线可视化")
    print("="*70)
    
    log_dir = "./logs_5000"
    
    convergence_data_raw = load_convergence_log(log_dir)
    history_data = load_training_history(log_dir)
    
    convergence_data = extract_convergence_data(convergence_data_raw)
    reward_data = extract_reward_data(history_data)
    
    print_statistics(convergence_data, reward_data)
    
    output_file = plot_convergence_curves(convergence_data, reward_data)
    
    print(f"\n请查看生成的图表: {output_file}")

if __name__ == "__main__":
    main()
