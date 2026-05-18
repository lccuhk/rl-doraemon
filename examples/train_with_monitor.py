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
        
        for _ in range(num_eval_episodes):
            rewards = self._run_episode(training=False)
            for i, r in enumerate(rewards):
                total_rewards[i] += r
                if r > 0:
                    wins[i] += 1
        
        avg_rewards = [r / num_eval_episodes for r in total_rewards]
        win_rates = [w / num_eval_episodes for w in wins]
        
        self.logger.info(f"评估结果 - 平均奖励: {avg_rewards}, 胜率: {win_rates}")
        
        self.training_history['wins'].append({
            'episode': episode,
            'win_rates': win_rates,
            'avg_rewards': avg_rewards
        })
        
        for agent in self.agents:
            if hasattr(agent, 'set_training'):
                agent.set_training(True)
        
        return [{
            'episode': episode,
            'win_rates': win_rates,
            'avg_rewards': avg_rewards,
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
    print("麻将AI - 实时训练监控")
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
        name="DQN_Mahjong_Realtime",
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
    
    trainer = RealtimeTrainer(
        env, agents, 
        log_dir="./logs_realtime", 
        checkpoint_dir="./checkpoints_realtime",
        web_port=8081,
        update_interval=5
    )
    
    num_episodes = 100
    print(f"\n开始训练 {num_episodes} 回合...")
    print(f"设备: {dqn_agent.device}")
    print(f"\n请在浏览器中打开 http://localhost:8081 查看实时训练进度")
    print("="*60 + "\n")
    
    history = trainer.train(
        num_episodes=num_episodes,
        save_freq=50,
        eval_freq=50,
        num_eval_episodes=10
    )
    
    print("\n训练完成！")
    print(f"总步数: {dqn_agent.steps_done}")
    print(f"最终epsilon: {dqn_agent.get_epsilon():.4f}")

if __name__ == "__main__":
    main()
