import numpy as np
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import os
import json
from datetime import datetime
import logging
import statistics
import signal
import sys

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

from ..environment.mahjong_env import MahjongEnv
from ..agents.base_agent import BaseAgent
from ..agents.random_agent import RandomAgent
from ..agents.dqn_agent import DQNAgent
from ..utils.logging import setup_logger

class SignalHandler:
    def __init__(self, trainer):
        self.trainer = trainer
        self.save_requested = False
        self.graceful_exit = False
        
    def handle_sigusr1(self, signum, frame):
        self.trainer.logger.info("="*60)
        self.trainer.logger.info("[信号] 收到 SIGUSR1，请求保存快照")
        self.trainer.logger.info("="*60)
        self.save_requested = True
        
    def handle_sigint(self, signum, frame):
        self.trainer.logger.info("="*60)
        self.trainer.logger.info("[信号] 收到 SIGINT (Ctrl+C)，准备优雅退出")
        self.trainer.logger.info("="*60)
        self.graceful_exit = True
        
    def handle_sigterm(self, signum, frame):
        self.trainer.logger.info("="*60)
        self.trainer.logger.info("[信号] 收到 SIGTERM，准备优雅退出")
        self.trainer.logger.info("="*60)
        self.graceful_exit = True

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
            'losses': [],
            'episode_details': []
        }
        
        self.episode_count = 0
        self.total_steps = 0
        
        self.training_stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'win_types': [],
            'action_distribution': {},
            'phase_distribution': {}
        }
    
    def train(self, num_episodes: int, save_freq: int = 100, 
              eval_freq: int = 100, num_eval_episodes: int = 10,
              snapshot_freq: int = 0, max_snapshots: int = 10,
              start_episode: int = 0):
        self.logger.info("="*60)
        self.logger.info("开始训练")
        self.logger.info("="*60)
        self.logger.info(f"总回合数: {num_episodes}")
        self.logger.info(f"起始回合: {start_episode + 1}")
        self.logger.info(f"检查点保存频率: 每 {save_freq} 回合")
        self.logger.info(f"评估频率: 每 {eval_freq} 回合")
        self.logger.info(f"评估回合数: {num_eval_episodes}")
        self.logger.info(f"Agent数量: {len(self.agents)}")
        if snapshot_freq > 0:
            self.logger.info(f"快照保存频率: 每 {snapshot_freq} 回合")
            self.logger.info(f"保留最近快照数: {max_snapshots}")
        else:
            self.logger.info(f"快照保存: 禁用")
        self.logger.info("="*60)
        
        self.signal_handler = SignalHandler(self)
        signal.signal(signal.SIGUSR1, self.signal_handler.handle_sigusr1)
        signal.signal(signal.SIGINT, self.signal_handler.handle_sigint)
        signal.signal(signal.SIGTERM, self.signal_handler.handle_sigterm)
        
        self.logger.info(f"进程ID: {os.getpid()}")
        self.logger.info("信号处理已注册:")
        self.logger.info("  - SIGUSR1: 手动触发保存快照")
        self.logger.info("  - SIGINT/SIGTERM: 优雅退出并保存")
        self.logger.info(f"使用方法: kill -SIGUSR1 {os.getpid()}")
        self.logger.info("="*60)
        
        self.snapshot_history = []
        
        for episode in tqdm(range(start_episode, num_episodes), desc="训练中"):
            self.episode_count = episode + 1
            
            self.logger.debug(f"--- 第 {episode + 1} 回合开始 ---")
            
            episode_rewards, episode_steps, episode_details = self._run_episode(training=True)
            
            self.total_steps += episode_steps
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_rewards)
            self.training_history['episode_details'].append({
                'episode': episode + 1,
                'steps': episode_steps,
                'rewards': episode_rewards,
                'winner': self.env.winner,
                'details': episode_details
            })
            
            self.training_stats['episode_rewards'].append(episode_rewards)
            self.training_stats['episode_lengths'].append(episode_steps)
            
            self.logger.info(f"第 {episode + 1} 回合结束: 步数={episode_steps}, 奖励={episode_rewards}")
            
            if self.env.winner >= 0:
                self.logger.info(f"  -> 获胜者: 玩家{self.env.winner}")
            else:
                self.logger.info(f"  -> 流局")
            
            if snapshot_freq > 0 and (episode + 1) % snapshot_freq == 0:
                self.logger.info(f"--- 保存快照 (第 {episode + 1} 回合) ---")
                self._save_episode_snapshot(episode, max_snapshots)
            
            if self.signal_handler.save_requested:
                self.logger.info("="*60)
                self.logger.info("[信号触发] 保存手动请求的快照")
                self.logger.info("="*60)
                self._save_episode_snapshot(episode, max_snapshots, manual=True)
                self.signal_handler.save_requested = False
            
            if (episode + 1) % 50 == 0:
                self._log_episode_stats()
            
            if (episode + 1) % save_freq == 0:
                self.logger.info(f"--- 保存检查点 (第 {episode + 1} 回合) ---")
                self._save_checkpoint(episode)
            
            if (episode + 1) % eval_freq == 0:
                self.logger.info(f"--- 开始评估 (第 {episode + 1} 回合后) ---")
                self._evaluate(episode, num_eval_episodes)
            
            if self.signal_handler.graceful_exit:
                self.logger.info("="*60)
                self.logger.info("[优雅退出] 保存当前状态后退出")
                self.logger.info("="*60)
                self._save_checkpoint(episode, is_final=False)
                self.logger.info(f"已在第 {episode + 1} 回合保存检查点")
                self.logger.info("训练已优雅终止")
                self.logger.info("="*60)
                return self.training_history
        
        self.logger.info("="*60)
        self.logger.info("训练完成！")
        self.logger.info(f"总回合数: {num_episodes}")
        self.logger.info(f"总步数: {self.total_steps}")
        self.logger.info("="*60)
        
        self._save_checkpoint(num_episodes, is_final=True)
        
        return self.training_history
    
    def _log_episode_stats(self):
        if not self.training_stats['episode_rewards']:
            return
        
        recent_rewards = self.training_stats['episode_rewards'][-50:]
        recent_lengths = self.training_stats['episode_lengths'][-50:]
        
        self.logger.info("="*60)
        self.logger.info(f"[回合统计] 第 {self.episode_count} 回合")
        self.logger.info("-"*60)
        
        if recent_rewards:
            player0_rewards = [r[0] for r in recent_rewards]
            self.logger.info(f"  最近50回合 DQN Agent 奖励:")
            self.logger.info(f"    平均: {statistics.mean(player0_rewards):.4f}")
            self.logger.info(f"    最小: {min(player0_rewards):.4f}")
            self.logger.info(f"    最大: {max(player0_rewards):.4f}")
            if len(player0_rewards) > 1:
                self.logger.info(f"    标准差: {statistics.stdev(player0_rewards):.4f}")
        
        if recent_lengths:
            self.logger.info(f"  最近50回合 步数:")
            self.logger.info(f"    平均: {statistics.mean(recent_lengths):.1f}")
            self.logger.info(f"    最小: {min(recent_lengths)}")
            self.logger.info(f"    最大: {max(recent_lengths)}")
        
        if self.training_stats['win_types']:
            self.logger.info(f"  和牌类型统计:")
            win_type_counts = {}
            for wt in self.training_stats['win_types']:
                win_type_counts[wt] = win_type_counts.get(wt, 0) + 1
            for wt, count in win_type_counts.items():
                self.logger.info(f"    {wt}: {count} 次")
        
        if self.training_stats['phase_distribution']:
            self.logger.info(f"  阶段分布:")
            total_phases = sum(self.training_stats['phase_distribution'].values())
            for phase, count in sorted(self.training_stats['phase_distribution'].items()):
                self.logger.info(f"    {phase}: {count} ({count/total_phases:.1%})")
        
        if self.training_stats['action_distribution']:
            top_actions = sorted(self.training_stats['action_distribution'].items(), 
                               key=lambda x: x[1], reverse=True)[:5]
            self.logger.info(f"  最常用动作 (Top 5):")
            total_actions = sum(self.training_stats['action_distribution'].values())
            for action, count in top_actions:
                self.logger.info(f"    动作 {action}: {count} 次 ({count/total_actions:.1%})")
        
        self.logger.info("="*60)
    
    def _run_episode(self, training: bool = True) -> Tuple[List[float], int, List[Dict]]:
        obs = self.env.reset()
        done = False
        episode_rewards = [0.0] * self.env.num_players
        episode_steps = 0
        episode_details = []
        
        for agent in self.agents:
            agent.reset()
        
        self.logger.debug(f"  环境重置完成，牌山剩余: {len(self.env.wall)}")
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            phase = obs.get('phase', 'unknown')
            
            self.training_stats['phase_distribution'][phase] = self.training_stats['phase_distribution'].get(phase, 0) + 1
            
            agent = self.agents[current_player]
            
            self.logger.debug(f"  步骤 {episode_steps}: 玩家{current_player}, 阶段={phase}, 合法动作={legal_actions}")
            
            action = agent.select_action(obs, legal_actions)
            
            self.training_stats['action_distribution'][action] = self.training_stats['action_distribution'].get(action, 0) + 1
            
            next_obs, reward, done, info = self.env.step(action)
            
            step_detail = {
                'step': episode_steps,
                'player': current_player,
                'phase': phase,
                'action': action,
                'reward': reward,
                'done': done
            }
            
            if 'win_type' in info:
                step_detail['win_type'] = info['win_type']
                self.training_stats['win_types'].append(info['win_type'])
                self.logger.info(f"  -> 和牌! 玩家{current_player}, {info['win_type']}")
                if 'win_info' in info:
                    step_detail['win_info'] = info['win_info']
                    self.logger.info(f"     番数: {info['win_info'].get('han', 0)}, 符数: {info['win_info'].get('fu', 0)}")
                    self.logger.info(f"     役种: {info['win_info'].get('yaku', [])}")
            
            episode_details.append(step_detail)
            
            if training:
                agent.observe(obs, action, reward, next_obs, done)
            
            episode_rewards[current_player] += reward
            obs = next_obs
            episode_steps += 1
        
        return episode_rewards, episode_steps, episode_details
    
    def _evaluate(self, episode: int, num_eval_episodes: int):
        self.logger.info("="*60)
        self.logger.info(f"开始评估 (第 {episode + 1} 回合后)")
        self.logger.info(f"评估回合数: {num_eval_episodes}")
        self.logger.info("="*60)
        
        for agent in self.agents:
            if hasattr(agent, 'set_training'):
                agent.set_training(False)
        
        total_rewards = [0.0] * self.env.num_players
        wins = [0] * self.env.num_players
        draws = 0
        actual_wins = [0] * self.env.num_players
        
        for eval_episode in range(num_eval_episodes):
            rewards, steps, details = self._run_episode(training=False)
            
            for i, r in enumerate(rewards):
                total_rewards[i] += r
                if r > 0:
                    wins[i] += 1
            
            if self.env.winner >= 0:
                actual_wins[self.env.winner] += 1
                self.logger.debug(f"  评估回合 {eval_episode + 1}: 玩家{self.env.winner}获胜, 步数={steps}, 奖励={rewards}")
            else:
                draws += 1
                self.logger.debug(f"  评估回合 {eval_episode + 1}: 流局, 步数={steps}, 奖励={rewards}")
        
        avg_rewards = [r / num_eval_episodes for r in total_rewards]
        win_rates = [w / num_eval_episodes for w in wins]
        actual_win_rates = [w / num_eval_episodes for w in actual_wins]
        draw_rate = draws / num_eval_episodes
        
        self.logger.info("-"*60)
        self.logger.info("评估结果汇总")
        self.logger.info("-"*60)
        self.logger.info(f"总评估回合: {num_eval_episodes}")
        self.logger.info(f"流局数: {draws} ({draw_rate*100:.1f}%)")
        self.logger.info(f"真正和牌数: {sum(actual_wins)} ({sum(actual_win_rates)*100:.1f}%)")
        self.logger.info("-"*60)
        
        for i in range(self.env.num_players):
            agent_type = "DQN" if i == 0 else "Random"
            self.logger.info(f"玩家{i} ({agent_type}):")
            self.logger.info(f"  平均奖励: {avg_rewards[i]:.4f}")
            self.logger.info(f"  正奖励率: {win_rates[i]*100:.1f}%")
            self.logger.info(f"  真正和牌率: {actual_win_rates[i]*100:.1f}%")
        
        self.logger.info("-"*60)
        self.logger.info(f"评估结果 - 平均奖励: {avg_rewards}, 正奖励率: {win_rates}")
        self.logger.info("="*60)
        
        self.training_history['wins'].append({
            'episode': episode,
            'win_rates': win_rates,
            'actual_win_rates': actual_win_rates,
            'avg_rewards': avg_rewards,
            'draw_rate': draw_rate,
            'draws': draws,
            'actual_wins': actual_wins
        })
        
        for agent in self.agents:
            if hasattr(agent, 'set_training'):
                agent.set_training(True)
    
    def _save_episode_snapshot(self, episode: int, max_snapshots: int = 10, manual: bool = False):
        snapshot_dir = os.path.join(self.checkpoint_dir, 'snapshots')
        os.makedirs(snapshot_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = "manual" if manual else "snapshot"
        snapshot_name = f"{prefix}_ep{episode+1}_{timestamp}"
        
        saved_files = []
        for i, agent in enumerate(self.agents):
            if hasattr(agent, 'save'):
                snapshot_path = os.path.join(snapshot_dir, f"{snapshot_name}_agent{i}.pt")
                agent.save(snapshot_path)
                saved_files.append(snapshot_path)
        
        if saved_files:
            if manual:
                self.logger.info(f"[手动保存] 快照已保存: {snapshot_name}")
            else:
                self.logger.debug(f"回合快照已保存: {snapshot_name}")
            
            if not manual:
                self.snapshot_history.append({
                    'episode': episode + 1,
                    'timestamp': timestamp,
                    'files': saved_files
                })
                
                while len(self.snapshot_history) > max_snapshots:
                    old_snapshot = self.snapshot_history.pop(0)
                    for file_path in old_snapshot['files']:
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                self.logger.debug(f"已删除旧快照: {os.path.basename(file_path)}")
                            except:
                                pass
    
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
            json.dump(self.training_history, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
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
