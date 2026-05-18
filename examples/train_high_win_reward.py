import sys
import os
import json
import time
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.training.trainer import Trainer

class RealtimeTrainer(Trainer):
    def __init__(self, env, agents, log_dir="./logs", checkpoint_dir="./checkpoints", 
                 web_port=8081, update_interval=10):
        super().__init__(env, agents, log_dir, checkpoint_dir)
        self.web_port = web_port
        self.update_interval = update_interval
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self._status_file = os.path.join(log_dir, 'realtime_status.json')
        
    def train(self, num_episodes, save_freq=100, eval_freq=100, num_eval_episodes=10):
        self.total_episodes = num_episodes
        self.logger.info(f"开始训练，共 {num_episodes} 个回合")
        
        self._save_status({
            'current_episode': 0,
            'total_episodes': num_episodes,
            'progress': 0,
            'is_training': True,
            'start_time': datetime.now().isoformat(),
            'elapsed_seconds': 0,
            'avg_reward': 0,
            'recent_rewards': [],
            'evaluation_results': []
        })
        
        start_time = datetime.now()
        
        for episode in range(num_episodes):
            self.current_episode = episode
            episode_rewards = self._run_episode(training=True)
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_rewards)
            
            if (episode + 1) % save_freq == 0:
                self._save_checkpoint(episode)
            
            if (episode + 1) % eval_freq == 0:
                eval_results = self._evaluate_realtime(episode, num_eval_episodes)
                self._save_status({
                    'current_episode': episode + 1,
                    'total_episodes': num_episodes,
                    'progress': ((episode + 1) / num_episodes) * 100,
                    'is_training': True,
                    'start_time': start_time.isoformat(),
                    'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
                    'avg_reward': sum(sum(r) for r in self.training_history['rewards'][-100:]) / max(1, min(100, len(self.training_history['rewards']))) / 4,
                    'recent_rewards': [sum(r) / 4 for r in self.training_history['rewards'][-20:]],
                    'evaluation_results': eval_results
                })
            elif (episode + 1) % self.update_interval == 0:
                self._save_status({
                    'current_episode': episode + 1,
                    'total_episodes': num_episodes,
                    'progress': ((episode + 1) / num_episodes) * 100,
                    'is_training': True,
                    'start_time': start_time.isoformat(),
                    'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
                    'avg_reward': sum(sum(r) for r in self.training_history['rewards'][-100:]) / max(1, min(100, len(self.training_history['rewards']))) / 4,
                    'recent_rewards': [sum(r) / 4 for r in self.training_history['rewards'][-20:]],
                    'evaluation_results': []
                })
        
        self._save_checkpoint(num_episodes, is_final=True)
        self.logger.info("训练完成！")
        
        self._save_status({
            'current_episode': num_episodes,
            'total_episodes': num_episodes,
            'progress': 100,
            'is_training': False,
            'start_time': start_time.isoformat(),
            'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
            'avg_reward': sum(sum(r) for r in self.training_history['rewards'][-100:]) / max(1, min(100, len(self.training_history['rewards']))) / 4,
            'recent_rewards': [sum(r) / 4 for r in self.training_history['rewards'][-20:]],
            'evaluation_results': []
        })
        
        return self.training_history
    
    def _evaluate_realtime(self, episode, num_eval_episodes):
        self.logger.info(f"第 {episode + 1} 回合后进行评估...")
        
        for agent in self.agents:
            if hasattr(agent, 'set_training'):
                agent.set_training(False)
        
        total_rewards = [0.0] * self.env.num_players
        wins = [0] * self.env.num_players
        actual_wins = [0] * self.env.num_players
        draws = 0
        
        for _ in range(num_eval_episodes):
            rewards = self._run_episode(training=False)
            for i, r in enumerate(rewards):
                total_rewards[i] += r
                if r > 0:
                    wins[i] += 1
                if r >= 10.0:
                    actual_wins[i] += 1
            if self.env.winner < 0:
                draws += 1
        
        avg_rewards = [r / num_eval_episodes for r in total_rewards]
        win_rates = [w / num_eval_episodes for w in wins]
        actual_win_rates = [w / num_eval_episodes for w in actual_wins]
        
        self.logger.info(f"评估结果 - 平均奖励: {avg_rewards}")
        self.logger.info(f"实际和牌率: {actual_win_rates}, 流局率: {draws/num_eval_episodes:.2%}")
        
        self.training_history['wins'].append({
            'episode': episode,
            'win_rates': win_rates,
            'actual_win_rates': actual_win_rates,
            'avg_rewards': avg_rewards
        })
        
        for agent in self.agents:
            if hasattr(agent, 'set_training'):
                agent.set_training(True)
        
        return [{
            'episode': episode,
            'win_rates': win_rates,
            'actual_win_rates': actual_win_rates,
            'avg_rewards': avg_rewards,
            'draw_rate': draws / num_eval_episodes,
            'timestamp': datetime.now().isoformat()
        }]
    
    def _save_status(self, status):
        try:
            os.makedirs(os.path.dirname(self._status_file), exist_ok=True)
            with open(self._status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass

def main():
    print("="*60)
    print("麻将AI - 高和牌奖励训练")
    print("="*60)
    
    print("\n【新奖励机制】")
    print("\n最终奖励（大幅增加和牌权重）:")
    print("  - 实际和牌: +15.0 (原 +1.0, 提升15倍)")
    print("  - 输家惩罚: -5.0 (原 -0.33, 提升15倍)")
    print("  - 流局最高分: +0.3 (原 +0.5)")
    print("  - 流局其他: -1.0 (原 -0.16)")
    
    print("\n中间奖励（降低权重，作为辅助）:")
    print("  - 向听数改进: +0.12/每向听 (原 +0.25)")
    print("  - 听牌(向听数=0): +0.25 (原 +0.5)")
    print("  - 立直: +0.15 (原 +0.3)")
    print("  - 杠: +0.1 (原 +0.2)")
    print("  - 碰: +0.08 (原 +0.15)")
    print("  - 吃: +0.05 (原 +0.1)")
    print("  - 打孤张字牌: +0.02 (原 +0.05)")
    print("  - 打孤张幺九: +0.01 (原 +0.03)")
    print("  - 向听数恶化: -0.05 (原 -0.1)")
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong_HighWinReward",
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
    
    trainer = RealtimeTrainer(
        env, agents, 
        log_dir="./logs_high_reward", 
        checkpoint_dir="./checkpoints_high_reward",
        web_port=8081,
        update_interval=5
    )
    
    num_episodes = 5000
    print(f"\n开始训练 {num_episodes} 回合...")
    print(f"设备: {dqn_agent.device}")
    print(f"\n请在浏览器中打开 http://localhost:8081 查看实时训练进度")
    print("="*60 + "\n")
    
    history = trainer.train(
        num_episodes=num_episodes,
        save_freq=50,
        eval_freq=50,
        num_eval_episodes=20
    )
    
    print("\n训练完成！")
    print(f"总步数: {dqn_agent.steps_done}")
    print(f"最终epsilon: {dqn_agent.get_epsilon():.4f}")

if __name__ == "__main__":
    main()
