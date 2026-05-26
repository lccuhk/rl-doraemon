import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import os
import logging

from .base_agent import BaseAgent
from ..models.dqn_model import DQNModel
from ..environment.mahjong_env import ActionType

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool, legal_actions: List[int]):
        self.buffer.append((state, action, reward, next_state, done, legal_actions))
    
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, legal_actions_list = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            legal_actions_list
        )
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgentDebug(BaseAgent):
    def __init__(self, name: str = "DQNAgentDebug", 
                 input_shape: Tuple[int, int, int] = (6, 34, 4),
                 num_actions: int = 41,
                 hidden_size: int = 256,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.9995,
                 batch_size: int = 64,
                 target_update_freq: int = 1000,
                 buffer_capacity: int = 100000,
                 device: Optional[str] = None,
                 seed: Optional[int] = None,
                 debug_level: int = 1):
        super().__init__(name)
        
        self.debug_level = debug_level
        self.logger = logging.getLogger('dqn_agent_debug')
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        self.policy_net = DQNModel(input_shape, num_actions, hidden_size).to(self.device)
        self.target_net = DQNModel(input_shape, num_actions, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.steps_done = 0
        self.episode_rewards = []
        
        self.training = True
        
        self.action_stats = {
            'total_actions': 0,
            'random_actions': 0,
            'greedy_actions': 0,
            'tsumo_available': 0,
            'tsumo_selected': 0,
            'ron_available': 0,
            'ron_selected': 0,
            'riichi_selected': 0,
            'pon_selected': 0,
            'chi_selected': 0,
            'kan_selected': 0,
            'pass_selected': 0,
            'discard_actions': 0
        }
        
        self.q_value_history = []
    
    def select_action(self, observation: Dict, legal_actions: List[int]) -> int:
        state = observation['obs']
        current_player = observation.get('player_id', -1)
        phase = observation.get('phase', -1)
        
        self.action_stats['total_actions'] += 1
        
        has_tsumo = ActionType.TSUMO.value in legal_actions
        has_ron = ActionType.RON.value in legal_actions
        
        if has_tsumo:
            self.action_stats['tsumo_available'] += 1
            if self.debug_level >= 1:
                self.logger.warning(f"[和牌机会] 玩家{current_player} 有自摸机会！合法动作: {legal_actions}")
        
        if has_ron:
            self.action_stats['ron_available'] += 1
            if self.debug_level >= 1:
                self.logger.warning(f"[和牌机会] 玩家{current_player} 有荣和机会！合法动作: {legal_actions}")
        
        if self.training and random.random() < self.epsilon:
            self.action_stats['random_actions'] += 1
            
            if has_tsumo or has_ron:
                if has_tsumo and has_ron:
                    win_action = random.choice([ActionType.TSUMO.value, ActionType.RON.value])
                elif has_tsumo:
                    win_action = ActionType.TSUMO.value
                else:
                    win_action = ActionType.RON.value
                
                if random.random() < 0.9:
                    action = win_action
                    if self.debug_level >= 1:
                        self.logger.warning(f"[随机-优先和牌] 玩家{current_player} 有和牌机会，优先选择和牌动作:{action}")
                else:
                    action = random.choice(legal_actions)
                    if self.debug_level >= 1:
                        if has_tsumo and action != ActionType.TSUMO.value:
                            self.logger.warning(f"[放弃自摸-随机] 玩家{current_player} 有自摸机会但随机选择了动作:{action}")
                        if has_ron and action != ActionType.RON.value:
                            self.logger.warning(f"[放弃荣和-随机] 玩家{current_player} 有荣和机会但随机选择了动作:{action}")
            else:
                if legal_actions:
                    action = random.choice(legal_actions)
                else:
                    action = 0
            
            if self.debug_level >= 2:
                self.logger.debug(f"[随机动作] 玩家{current_player} 阶段:{phase} "
                                f"epsilon:{self.epsilon:.4f} 选择动作:{action}")
            
            self._update_action_stats(action)
            return action
        
        self.action_stats['greedy_actions'] += 1
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            legal_mask = torch.ones(self.num_actions) * float('-inf')
            for la in legal_actions:
                if 0 <= la < self.num_actions:
                    legal_mask[la] = q_values[0, la]
            
            action = legal_mask.argmax().item()
            
            if self.debug_level >= 2:
                q_values_np = q_values[0].cpu().numpy()
                legal_q_values = [(a, q_values_np[a]) for a in legal_actions if 0 <= a < self.num_actions]
                legal_q_values.sort(key=lambda x: x[1], reverse=True)
                top_q = legal_q_values[:5] if len(legal_q_values) > 5 else legal_q_values
                self.logger.debug(f"[Q值选择] 玩家{current_player} 阶段:{phase} "
                                f"选择动作:{action} Q值:{q_values_np[action]:.4f} "
                                f"Top5:{[(a, f'{q:.4f}') for a, q in top_q]}")
            
            if has_tsumo:
                tsumo_q = q_values_np[ActionType.TSUMO.value]
                selected_q = q_values_np[action]
                if action == ActionType.TSUMO.value:
                    self.action_stats['tsumo_selected'] += 1
                    self.logger.warning(f"[选择自摸] 玩家{current_player} 自摸Q值:{tsumo_q:.4f} 是最优选择")
                else:
                    self.logger.warning(f"[放弃自摸-贪心] 玩家{current_player} "
                                      f"自摸Q值:{tsumo_q:.4f} < 选择动作{action}的Q值:{selected_q:.4f}")
            
            if has_ron:
                ron_q = q_values_np[ActionType.RON.value]
                selected_q = q_values_np[action]
                if action == ActionType.RON.value:
                    self.action_stats['ron_selected'] += 1
                    self.logger.warning(f"[选择荣和] 玩家{current_player} 荣和Q值:{ron_q:.4f} 是最优选择")
                else:
                    self.logger.warning(f"[放弃荣和-贪心] 玩家{current_player} "
                                      f"荣和Q值:{ron_q:.4f} < 选择动作{action}的Q值:{selected_q:.4f}")
            
            self.q_value_history.append({
                'step': self.steps_done,
                'max_q': q_values.max().item(),
                'min_q': q_values.min().item(),
                'mean_q': q_values.mean().item(),
                'selected_q': q_values[0, action].item() if 0 <= action < self.num_actions else 0
            })
        
        self._update_action_stats(action)
        return action
    
    def _update_action_stats(self, action: int):
        if action == ActionType.RIICHI.value:
            self.action_stats['riichi_selected'] += 1
        elif action == ActionType.PON.value:
            self.action_stats['pon_selected'] += 1
        elif action == ActionType.CHI.value:
            self.action_stats['chi_selected'] += 1
        elif action == ActionType.KAN.value:
            self.action_stats['kan_selected'] += 1
        elif action == ActionType.PASS.value:
            self.action_stats['pass_selected'] += 1
        elif 0 <= action < 34:
            self.action_stats['discard_actions'] += 1
    
    def observe(self, observation: Dict, action: int, reward: float, 
                next_observation: Dict, done: bool):
        if not self.training:
            return
        
        state = observation['obs']
        next_state = next_observation['obs']
        legal_actions = observation['legal_actions']
        
        if self.debug_level >= 2 and abs(reward) > 0.1:
            self.logger.debug(f"[观察] 动作:{action} 奖励:{reward:.4f} 结束:{done}")
        
        self.replay_buffer.push(state, action, reward, next_state, done, legal_actions)
        
        if len(self.replay_buffer) >= self.batch_size:
            loss = self._train_step()
            if self.debug_level >= 2 and loss is not None:
                self.logger.debug(f"[训练] Loss:{loss:.6f}")
        
        self.steps_done += 1
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            if self.debug_level >= 1:
                self.logger.info(f"[目标网络更新] 步数:{self.steps_done}")
        
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
    
    def _train_step(self):
        states, actions, rewards, next_states, dones, legal_actions_list = \
            self.replay_buffer.sample(self.batch_size)
        
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_states_tensor = torch.FloatTensor(next_states).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)
        
        current_q_values = self.policy_net(states_tensor).gather(1, actions_tensor.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_net(next_states_tensor)
            
            for i in range(self.batch_size):
                legal = legal_actions_list[i]
                mask = torch.ones(self.num_actions) * float('-inf')
                for la in legal:
                    if 0 <= la < self.num_actions:
                        mask[la] = next_q_values[i, la]
                next_q_values[i] = mask
            
            max_next_q_values = next_q_values.max(1)[0]
            expected_q_values = rewards_tensor + (1 - dones_tensor) * self.gamma * max_next_q_values
        
        loss = self.criterion(current_q_values.squeeze(), expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path: str):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'action_stats': self.action_stats,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.action_stats = checkpoint.get('action_stats', self.action_stats)
            print(f"Model loaded from {path}")
        else:
            print(f"Model file not found: {path}")
    
    def set_training(self, training: bool):
        self.training = training
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()
    
    def reset(self):
        pass
    
    def get_epsilon(self) -> float:
        return self.epsilon
    
    def get_action_stats(self) -> Dict:
        return self.action_stats.copy()
    
    def print_action_stats(self):
        stats = self.action_stats
        self.logger.info("="*60)
        self.logger.info("DQN Agent 动作统计")
        self.logger.info("="*60)
        self.logger.info(f"总动作数: {stats['total_actions']}")
        self.logger.info(f"  - 随机动作: {stats['random_actions']} ({stats['random_actions']/max(1, stats['total_actions'])*100:.1f}%)")
        self.logger.info(f"  - 贪心动作: {stats['greedy_actions']} ({stats['greedy_actions']/max(1, stats['total_actions'])*100:.1f}%)")
        self.logger.info("")
        self.logger.info(f"自摸机会: {stats['tsumo_available']}, 选择自摸: {stats['tsumo_selected']}")
        self.logger.info(f"荣和机会: {stats['ron_available']}, 选择荣和: {stats['ron_selected']}")
        self.logger.info("")
        self.logger.info(f"立直: {stats['riichi_selected']}")
        self.logger.info(f"碰: {stats['pon_selected']}")
        self.logger.info(f"吃: {stats['chi_selected']}")
        self.logger.info(f"杠: {stats['kan_selected']}")
        self.logger.info(f"PASS: {stats['pass_selected']}")
        self.logger.info(f"舍牌: {stats['discard_actions']}")
        self.logger.info("="*60)
