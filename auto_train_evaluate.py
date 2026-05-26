import sys
import os
import subprocess
import time
import glob
import argparse
import re
import json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class ConvergenceLogger:
    def __init__(self, log_dir="./logs_5000"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.convergence_log = os.path.join(log_dir, f"convergence_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.history = {
            'start_time': datetime.now().isoformat(),
            'convergence_events': [],
            'training_stats': []
        }
        self.last_loss = None
        self.last_reward = None
        self.last_epsilon = None
        self.consecutive_stable = 0
        self.stable_threshold = 3
    
    def parse_training_stats(self, log_content):
        stats = {}
        
        training_stats_match = re.search(r'\[训练统计\](.*?)(?=\[训练统计\]|$)', log_content, re.DOTALL)
        
        if not training_stats_match:
            return stats
        
        training_stats_block = training_stats_match.group(1)
        
        patterns = {
            'steps': r'步数=(\d+)',
            'avg_loss': r'最近100次损失:.*?平均:\s+([\d.]+)',
            'min_loss': r'最近100次损失:.*?最小:\s+([\d.]+)',
            'max_loss': r'最近100次损失:.*?最大:\s+([\d.]+)',
            'loss_std': r'最近100次损失:.*?标准差:\s+([\d.]+)',
            'avg_reward': r'最近1000次奖励:.*?平均:\s+([\d.-]+)',
            'min_reward': r'最近1000次奖励:.*?最小:\s+([\d.-]+)',
            'max_reward': r'最近1000次奖励:.*?最大:\s+([\d.-]+)',
            'reward_std': r'最近1000次奖励:.*?标准差:\s+([\d.]+)',
            'epsilon': r'探索率:\s+([\d.]+)%',
            'exploration_count': r'探索次数:\s+(\d+)',
            'exploitation_count': r'利用次数:\s+(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, training_stats_block, re.DOTALL)
            if match:
                try:
                    value = match.group(1)
                    stats[key] = float(value) if '.' in value else int(value)
                except:
                    pass
        
        return stats
    
    def check_convergence(self, current_stats):
        if not current_stats or 'avg_loss' not in current_stats:
            return False, None
        
        if self.last_loss is None:
            self.last_loss = current_stats.get('avg_loss')
            self.last_reward = current_stats.get('avg_reward')
            self.last_epsilon = current_stats.get('epsilon')
            return False, None
        
        current_loss = current_stats.get('avg_loss')
        current_reward = current_stats.get('avg_reward')
        current_epsilon = current_stats.get('epsilon')
        
        loss_change = abs(current_loss - self.last_loss) / self.last_loss if self.last_loss > 0 else 0
        reward_change = abs(current_reward - self.last_reward) / abs(self.last_reward) if self.last_reward != 0 else 0
        
        convergence_threshold = 0.05
        epsilon_threshold = 2.0
        
        is_stable = (
            loss_change < convergence_threshold and
            current_epsilon <= epsilon_threshold
        )
        
        if is_stable:
            self.consecutive_stable += 1
        else:
            self.consecutive_stable = 0
        
        is_converged = self.consecutive_stable >= self.stable_threshold
        
        convergence_info = {
            'timestamp': datetime.now().isoformat(),
            'steps': current_stats.get('steps'),
            'current_loss': current_loss,
            'previous_loss': self.last_loss,
            'loss_change_pct': loss_change * 100,
            'current_reward': current_reward,
            'previous_reward': self.last_reward,
            'reward_change_pct': reward_change * 100,
            'epsilon': current_epsilon,
            'consecutive_stable': self.consecutive_stable,
            'is_converged': is_converged,
            'loss_std': current_stats.get('loss_std'),
            'reward_std': current_stats.get('reward_std'),
            'exploration_count': current_stats.get('exploration_count'),
            'exploitation_count': current_stats.get('exploitation_count')
        }
        
        self.last_loss = current_loss
        self.last_reward = current_reward
        self.last_epsilon = current_epsilon
        
        return is_converged, convergence_info
    
    def log_stats(self, stats, convergence_info=None):
        entry = {
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }
        if convergence_info:
            entry['convergence_info'] = convergence_info
        
        self.history['training_stats'].append(entry)
        
        if convergence_info and convergence_info.get('is_converged'):
            self.history['convergence_events'].append({
                'timestamp': convergence_info['timestamp'],
                'steps': convergence_info['steps'],
                'loss': convergence_info['current_loss'],
                'reward': convergence_info['current_reward'],
                'epsilon': convergence_info['epsilon']
            })
        
        self.save()
    
    def save(self):
        with open(self.convergence_log, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
    
    def print_convergence_report(self):
        print("\n" + "="*70)
        print("收敛检测报告")
        print("="*70)
        
        if not self.history['convergence_events']:
            print("尚未检测到收敛事件")
            return
        
        print(f"检测到 {len(self.history['convergence_events'])} 次收敛事件:")
        for i, event in enumerate(self.history['convergence_events'], 1):
            print(f"\n事件 {i}:")
            print(f"  时间: {event['timestamp']}")
            print(f"  步数: {event['steps']}")
            print(f"  损失: {event['loss']:.6f}")
            print(f"  奖励: {event['reward']:.4f}")
            print(f"  探索率: {event['epsilon']:.2f}%")
        
        print("="*70)

def find_training_log():
    log_files = sorted(glob.glob('training_5000_*.log'))
    if log_files:
        return log_files[-1]
    return None

def monitor_training_with_convergence(training_pid, check_interval=60):
    print("\n监控模式: 启动收敛检测...")
    print("="*70)
    
    convergence_logger = ConvergenceLogger()
    training_log = find_training_log()
    
    if training_log:
        print(f"监控训练日志: {training_log}")
    else:
        print("警告: 未找到训练日志文件")
    
    last_position = 0
    last_stats_time = 0
    
    print(f"收敛检测配置:")
    print(f"  - 损失变化阈值: 5%")
    print(f"  - 探索率阈值: 2.0%")
    print(f"  - 连续稳定次数: 3 次")
    print("="*70)
    
    while True:
        result = subprocess.run(
            ['kill', '-0', str(training_pid)],
            capture_output=True
        )
        if result.returncode != 0:
            break
        
        if training_log and os.path.exists(training_log):
            current_size = os.path.getsize(training_log)
            
            if current_size > last_position:
                with open(training_log, 'r', encoding='utf-8') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    last_position = current_size
                
                if '[训练统计]' in new_content:
                    current_time = time.time()
                    if current_time - last_stats_time >= check_interval:
                        stats = convergence_logger.parse_training_stats(new_content)
                        
                        if stats:
                            is_converged, convergence_info = convergence_logger.check_convergence(stats)
                            convergence_logger.log_stats(stats, convergence_info)
                            
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 训练统计:")
                            print(f"  步数: {stats.get('steps', 'N/A')}")
                            print(f"  平均损失: {stats.get('avg_loss', 'N/A'):.6f}")
                            print(f"  平均奖励: {stats.get('avg_reward', 'N/A'):.4f}")
                            print(f"  探索率: {stats.get('epsilon', 'N/A'):.2f}%")
                            
                            if is_converged:
                                print(f"\n  🎯 检测到收敛!")
                                print(f"     连续稳定次数: {convergence_info['consecutive_stable']}")
                                print(f"     损失变化: {convergence_info['loss_change_pct']:.2f}%")
                                convergence_logger.print_convergence_report()
                        
                        last_stats_time = current_time
        
        time.sleep(check_interval)
    
    convergence_logger.history['end_time'] = datetime.now().isoformat()
    convergence_logger.save()
    
    print("\n" + "="*70)
    print("训练完成！最终收敛报告:")
    convergence_logger.print_convergence_report()
    print(f"收敛日志已保存到: {convergence_logger.convergence_log}")
    print("="*70)
    
    return 0

def run_training(episodes=5000, resume=True):
    print("="*70)
    print("开始训练...")
    print("="*70)
    
    cmd = [sys.executable, "train_5000_episodes.py"]
    if resume:
        cmd.append("--resume")
    cmd.extend(["--episodes", str(episodes)])
    
    log_file = f"training_auto_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    print(f"训练命令: {' '.join(cmd)}")
    print(f"日志文件: {log_file}")
    print("="*70)
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )
    
    return process, log_file

def wait_for_training_completion(process, log_file, check_interval=60):
    print("\n等待训练完成...")
    print("="*70)
    
    last_size = 0
    no_change_count = 0
    max_no_change = 10
    
    while True:
        if process.poll() is not None:
            break
        
        if os.path.exists(log_file):
            current_size = os.path.getsize(log_file)
            if current_size == last_size:
                no_change_count += 1
                if no_change_count >= max_no_change:
                    print(f"警告: 日志文件 {max_no_change * check_interval} 秒未更新，可能训练已停滞")
            else:
                no_change_count = 0
            last_size = current_size
        
        time.sleep(check_interval)
    
    return process.returncode

def find_latest_model(checkpoint_dir="./checkpoints_5000"):
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

def run_evaluation(model_path, episodes=100, compare=True):
    print("\n" + "="*70)
    print("训练完成！开始评估最终模型...")
    print("="*70)
    
    cmd = [
        sys.executable, "evaluate_final_model.py",
        "--model", model_path,
        "--episodes", str(episodes),
        "--checkpoint-dir", "./checkpoints_5000"
    ]
    
    if compare:
        cmd.append("--compare")
    
    print(f"评估命令: {' '.join(cmd)}")
    print("="*70)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("错误信息:")
        print(result.stderr)
    
    return result.returncode

def save_evaluation_report(eval_output, log_dir="./logs_5000"):
    os.makedirs(log_dir, exist_ok=True)
    report_file = os.path.join(log_dir, f"evaluation_report_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("麻将AI模型评估报告\n")
        f.write("="*70 + "\n")
        f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(eval_output)
    
    print(f"\n评估报告已保存到: {report_file}")
    return report_file

def main():
    parser = argparse.ArgumentParser(description='麻将AI - 训练与评估自动化流程')
    parser.add_argument('--episodes', type=int, default=5000, help='总训练回合数（默认5000）')
    parser.add_argument('--eval-episodes', type=int, default=100, help='评估回合数（默认100）')
    parser.add_argument('--no-resume', action='store_true', help='不恢复训练，从头开始')
    parser.add_argument('--no-compare', action='store_true', help='不与随机策略对比')
    parser.add_argument('--monitor-only', action='store_true', help='只监控现有训练进程，不启动新训练')
    args = parser.parse_args()
    
    print("="*70)
    print("麻将AI - 训练与评估自动化流程")
    print("="*70)
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"训练回合数: {args.episodes}")
    print(f"评估回合数: {args.eval_episodes}")
    print(f"恢复训练: {'否' if args.no_resume else '是'}")
    print(f"与随机策略对比: {'否' if args.no_compare else '是'}")
    print("="*70)
    
    training_process = None
    log_file = None
    training_pid = None
    
    if args.monitor_only:
        print("\n监控模式: 查找现有训练进程...")
        
        result = subprocess.run(
            ['pgrep', '-f', 'train_5000_episodes.py'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            training_pid = int(pids[0])
            print(f"发现训练进程: PID={training_pid}")
        else:
            print("未找到运行中的训练进程")
            return
    else:
        training_process, log_file = run_training(
            episodes=args.episodes,
            resume=not args.no_resume
        )
    
    try:
        if args.monitor_only:
            return_code = monitor_training_with_convergence(training_pid)
        else:
            return_code = wait_for_training_completion(training_process, log_file)
        
        if return_code == 0:
            print("\n✅ 训练成功完成！")
            
            print("\n查找最新模型...")
            model_path, model_type = find_latest_model()
            
            if model_path:
                print(f"找到模型: {model_path}")
                print(f"模型类型: {model_type}")
                
                eval_return_code = run_evaluation(
                    model_path=model_path,
                    episodes=args.eval_episodes,
                    compare=not args.no_compare
                )
                
                if eval_return_code == 0:
                    print("\n✅ 评估完成！")
                else:
                    print(f"\n❌ 评估失败，返回码: {eval_return_code}")
            else:
                print("\n❌ 未找到模型文件")
        else:
            print(f"\n❌ 训练失败，返回码: {return_code}")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        if training_process and not args.monitor_only:
            print("正在停止训练...")
            training_process.terminate()
            try:
                training_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                training_process.kill()
            print("训练已停止")
    
    print("\n" + "="*70)
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

if __name__ == "__main__":
    main()
