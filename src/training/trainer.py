import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import os
import json
from datetime import datetime

from ..environment.mahjong_env import MahjongEnv
from ..agents.base_agent import BaseAgent
from ..agents.random_agent import RandomAgent
from ..agents.dqn_agent import DQNAgent
from ..utils.logging import setup_logger

class Trainer:
    def __init__(self, env: MahjongEnv, agents: List[BaseAgent], 
                 log_dir: str = "./logs", checkpoint_dir: str = "./checkpoints"):
        self.env = env
        self.agents = agents
        self.logger = setup_logger('mahjong_trainer')
        
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'wins': [],
            'losses': []
        }
    
    def train(self, num_episodes: int, save_freq: int = 100, 
              eval_freq: int = 100, num_eval_episodes: int = 10):
        self.logger.info(f"开始训练，共 {num_episodes} 个回合")
        
        for episode in tqdm(range(num_episodes), desc="训练中"):
            episode_rewards = self._run_episode(training=True)
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_rewards)
            
            if (episode + 1) % save_freq == 0:
                self._save_checkpoint(episode)
            
            if (episode + 1) % eval_freq == 0:
                self._evaluate(episode, num_eval_episodes)
        
        self._save_checkpoint(num_episodes, is_final=True)
        self.logger.info("训练完成！")
        
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
        
        return episode_rewards
    
    def _evaluate(self, episode: int, num_eval_episodes: int):
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
    
    def _save_checkpoint(self, episode: int, is_final: bool = False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if is_final:
            filename = f"final_model_{timestamp}.pt"
        else:
            filename = f"checkpoint_{episode}_{timestamp}.pt"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'save'):
                agent_path = filepath.replace('.pt', f'_agent{i}.pt')
                agent.save(agent_path)
        
        history_path = os.path.join(self.log_dir, f"training_history_{timestamp}.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"检查点已保存: {filepath}")
    
    def self_play(self, num_episodes: int, agent: BaseAgent, 
                  opponent: Optional[BaseAgent] = None):
        if opponent is None:
            opponent = RandomAgent("Opponent")
        
        self.agents = [agent, opponent, opponent, opponent]
        
        return self.train(num_episodes)
    
    def play_game(self, render: bool = True) -> Tuple[int, List[int]]:
        obs = self.env.reset()
        done = False
        
        for agent in self.agents:
            agent.reset()
        
        while not done:
            current_player = obs['current_player']
            
            if render:
                self.env.render(current_player)
            
            legal_actions = obs['legal_actions']
            agent = self.agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            obs, reward, done, info = self.env.step(action)
        
        winner = self.env.winner
        final_scores = [p['score'] for p in self.env.players]
        
        if render:
            print(f"\n游戏结束！")
            if winner >= 0:
                print(f"获胜者: 玩家 {winner}")
            else:
                print("流局")
            print(f"最终分数: {final_scores}")
        
        return winner, final_scores
    
    def tournament(self, num_games: int = 100) -> Dict:
        results = {
            'wins': [0] * self.env.num_players,
            'draws': 0,
            'total_games': num_games,
            'scores': []
        }
        
        for _ in tqdm(range(num_games), desc="锦标赛中"):
            winner, scores = self.play_game(render=False)
            results['scores'].append(scores)
            
            if winner >= 0:
                results['wins'][winner] += 1
            else:
                results['draws'] += 1
        
        results['win_rates'] = [w / num_games for w in results['wins']]
        results['avg_scores'] = [
            sum(s[i] for s in results['scores']) / num_games 
            for i in range(self.env.num_players)
        ]
        
        return results
