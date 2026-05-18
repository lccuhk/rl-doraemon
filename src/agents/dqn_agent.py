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

class DQNAgent(BaseAgent):
    def __init__(self, name: str = "DQNAgent", 
                 input_shape: Tuple[int, int, int] = (6, 34, 4),
                 num_actions: int = 38,
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
                 seed: Optional[int] = None):
        super().__init__(name)
        
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
        
        self.logger = logging.getLogger('dqn_agent')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def select_action(self, observation: Dict, legal_actions: List[int]) -> int:
        state = observation['obs']
        player_id = observation.get('player_id', -1)
        
        has_tsumo = ActionType.TSUMO.value in legal_actions
        has_ron = ActionType.RON.value in legal_actions
        
        if has_tsumo or has_ron:
            self.logger.warning("="*60)
            self.logger.warning(f"[和牌机会] 玩家{player_id}")
            self.logger.warning(f"  合法动作: {legal_actions}")
            self.logger.warning(f"  自摸可用: {has_tsumo} (TSUMO={ActionType.TSUMO.value})")
            self.logger.warning(f"  荣和可用: {has_ron} (RON={ActionType.RON.value})")
            self.logger.warning(f"  当前epsilon: {self.epsilon:.4f}")
        
        if self.training and random.random() < self.epsilon:
            if has_tsumo or has_ron:
                if has_tsumo and has_ron:
                    win_action = random.choice([ActionType.TSUMO.value, ActionType.RON.value])
                elif has_tsumo:
                    win_action = ActionType.TSUMO.value
                else:
                    win_action = ActionType.RON.value
                
                if random.random() < 0.9:
                    self.logger.warning(f"[优先和牌-随机] 玩家{player_id} 选择动作: {win_action} "
                                      f"({'自摸' if win_action == ActionType.TSUMO.value else '荣和'})")
                    self.logger.warning("="*60)
                    return win_action
                else:
                    self.logger.warning(f"[放弃和牌-探索] 玩家{player_id} 继续随机探索")
            
            if legal_actions:
                action = random.choice(legal_actions)
                if has_tsumo or has_ron:
                    self.logger.warning(f"[放弃和牌-随机] 玩家{player_id} 随机选择了动作: {action}")
                    self.logger.warning("="*60)
                return action
            return 0
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            legal_mask = torch.ones(self.num_actions) * float('-inf')
            for action in legal_actions:
                if 0 <= action < self.num_actions:
                    legal_mask[action] = q_values[0, action]
            
            action = legal_mask.argmax().item()
            
            if has_tsumo or has_ron:
                q_values_np = q_values[0].cpu().numpy()
                tsumo_q = q_values_np[ActionType.TSUMO.value] if has_tsumo else float('-inf')
                ron_q = q_values_np[ActionType.RON.value] if has_ron else float('-inf')
                selected_q = q_values_np[action] if 0 <= action < self.num_actions else float('-inf')
                
                self.logger.warning(f"[贪心决策] 玩家{player_id}")
                self.logger.warning(f"  自摸Q值: {tsumo_q:.4f}")
                self.logger.warning(f"  荣和Q值: {ron_q:.4f}")
                self.logger.warning(f"  选择动作: {action}, Q值: {selected_q:.4f}")
                
                if action == ActionType.TSUMO.value:
                    self.logger.warning(f"  -> 选择自摸!")
                elif action == ActionType.RON.value:
                    self.logger.warning(f"  -> 选择荣和!")
                else:
                    self.logger.warning(f"  -> 放弃和牌，选择其他动作")
                self.logger.warning("="*60)
            
            return action
    
    def observe(self, observation: Dict, action: int, reward: float, 
                next_observation: Dict, done: bool):
        if not self.training:
            return
        
        state = observation['obs']
        next_state = next_observation['obs']
        legal_actions = observation['legal_actions']
        
        self.replay_buffer.push(state, action, reward, next_state, done, legal_actions)
        
        if len(self.replay_buffer) >= self.batch_size:
            self._train_step()
        
        self.steps_done += 1
        
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
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
                for action in legal:
                    if 0 <= action < self.num_actions:
                        mask[action] = next_q_values[i, action]
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
