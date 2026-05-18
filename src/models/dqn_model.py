import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DQNModel(nn.Module):
    def __init__(self, input_shape: Tuple[int, int, int] = (6, 34, 4), 
                 num_actions: int = 38, hidden_size: int = 256):
        super(DQNModel, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        self.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.dropout = nn.Dropout(0.3)
        
        conv_output_size = self._get_conv_output_size()
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_actions)
        
        self._initialize_weights()
    
    def _get_conv_output_size(self) -> int:
        dummy_input = torch.zeros(1, *self.input_shape)
        with torch.no_grad():
            x = F.relu(self.bn1(self.conv1(dummy_input)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            return x.numel()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def get_action_values(self, x: torch.Tensor, legal_actions: torch.Tensor = None) -> torch.Tensor:
        q_values = self.forward(x)
        
        if legal_actions is not None:
            mask = torch.ones_like(q_values) * float('-inf')
            for i in range(q_values.size(0)):
                legal = legal_actions[i]
                for action in legal:
                    if 0 <= action < self.num_actions:
                        mask[i, action] = q_values[i, action]
            return mask
        
        return q_values
