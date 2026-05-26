import sys
import os
import json
import time
import threading
from datetime import datetime
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env_debug import MahjongEnvDebug
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent_debug import DQNAgentDebug
from src.training.trainer import Trainer

class DebugTrainer(Trainer):
    def __init__(self, env, agents, log_dir="./logs", checkpoint_dir="./checkpoints", 
                 web_port=8081, update_interval=10):
        super().__init__(env, agents, log_dir, checkpoint_dir)
        self.web_port = web_port
        self.update_interval = update_interval
        self.current_episode = 0
        self.total_episodes = 0
        self.episode_rewards = []
        self._status_file = os.path.join(log_dir, 'realtime_status.json')
        
        self.debug_stats = {
            'total_episodes': 0,
            'actual_wins': 0,
            'draws': 0,
            'tsumo_opportunities': 0,
            'ron_opportunities': 0,
            'tsumo_taken': 0,
            'ron_taken': 0,
            'tsumo_missed_random': 0,
            'tsumo_missed_greedy': 0,
            'ron_missed_random': 0,
            'ron_missed_greedy': 0
        }
        
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
            'evaluation_results': [],
            'debug_stats': self.debug_stats
        })
        
        start_time = datetime.now()
        
        for episode in range(num_episodes):
            self.current_episode = episode
            episode_rewards = self._run_episode(training=True)
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_rewards)
            
            if (episode + 1) % save_freq == 0:
                self._save_checkpoint(episode)
                self._print_debug_summary()
            
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
                    'evaluation_results': eval_results,
                    'debug_stats': self.debug_stats
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
                    'evaluation_results': [],
                    'debug_stats': self.debug_stats
                })
        
        self._save_checkpoint(num_episodes, is_final=True)
        self.logger.info("训练完成！")
        self._print_debug_summary()
        
        self._save_status({
            'current_episode': num_episodes,
            'total_episodes': num_episodes,
            'progress': 100,
            'is_training': False,
            'start_time': start_time.isoformat(),
            'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
            'avg_reward': sum(sum(r) for r in self.training_history['rewards'][-100:]) / max(1, min(100, len(self.training_history['rewards']))) / 4,
            'recent_rewards': [sum(r) / 4 for r in self.training_history['rewards'][-20:]],
            'evaluation_results': [],
            'debug_stats': self.debug_stats
        })
        
        return self.training_history
    
    def _run_episode(self, training: bool = True) -> List[float]:
        obs = self.env.reset()
        done = False
        episode_rewards = [0.0] * self.env.num_players
        
        for agent in self.agents:
            agent.reset()
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            
            agent = self.agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = self.env.step(action)
            
            if training:
                agent.observe(obs, action, reward, next_obs, done)
            
            episode_rewards[current_player] += reward
            obs = next_obs
        
        self.debug_stats['total_episodes'] += 1
        if self.env.winner >= 0:
            self.debug_stats['actual_wins'] += 1
        else:
            self.debug_stats['draws'] += 1
        
        return episode_rewards
    
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
    
    def _print_debug_summary(self):
        self.logger.info("\n" + "="*70)
        self.logger.info("调试统计摘要")
        self.logger.info("="*70)
        
        stats = self.debug_stats
        total = stats['total_episodes']
        
        self.logger.info(f"总回合数: {total}")
        self.logger.info(f"  - 实际和牌: {stats['actual_wins']} ({stats['actual_wins']/max(1, total)*100:.1f}%)")
        self.logger.info(f"  - 流局: {stats['draws']} ({stats['draws']/max(1, total)*100:.1f}%)")
        
        for agent in self.agents:
            if hasattr(agent, 'print_action_stats'):
                agent.print_action_stats()
        
        self.logger.info("="*70 + "\n")
    
    def _save_status(self, status):
        try:
            os.makedirs(os.path.dirname(self._status_file), exist_ok=True)
            with open(self._status_file, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass

def main():
    print("="*70)
    print("麻将AI - 详细日志调试训练")
    print("="*70)
    
    print("\n【调试日志说明】")
    print("  - [自摸机会] / [荣和机会]: 检测到和牌机会")
    print("  - [选择自摸] / [选择荣和]: 模型正确选择了和牌动作")
    print("  - [放弃自摸-随机] / [放弃荣和-随机]: 随机探索时错过了和牌")
    print("  - [放弃自摸-贪心] / [放弃荣和-贪心]: 模型认为其他动作Q值更高")
    print("  - [Q值选择]: 显示贪心选择时的Q值对比")
    
    print("\n【新奖励机制】")
    print("  - 实际和牌: +15.0 (原 +1.0, 提升15倍)")
    print("  - 输家惩罚: -5.0 (原 -0.33, 提升15倍)")
    
    env = MahjongEnvDebug(seed=42, use_intermediate_rewards=True, debug_level=1)
    
    dqn_agent = DQNAgentDebug(
        name="DQN_Mahjong_Debug",
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
        seed=42,
        debug_level=1
    )
    
    opponent = RandomAgent("Opponent", seed=43)
    agents = [dqn_agent, opponent, opponent, opponent]
    
    trainer = DebugTrainer(
        env, agents, 
        log_dir="./logs_debug", 
        checkpoint_dir="./checkpoints_debug",
        web_port=8081,
        update_interval=5
    )
    
    num_episodes = 100
    print(f"\n开始训练 {num_episodes} 回合...")
    print(f"设备: {dqn_agent.device}")
    print(f"\n请在浏览器中打开 http://localhost:8081 查看实时训练进度")
    print("="*70 + "\n")
    
    history = trainer.train(
        num_episodes=num_episodes,
        save_freq=25,
        eval_freq=25,
        num_eval_episodes=10
    )
    
    print("\n训练完成！")
    print(f"总步数: {dqn_agent.steps_done}")
    print(f"最终epsilon: {dqn_agent.get_epsilon():.4f}")

if __name__ == "__main__":
    main()
