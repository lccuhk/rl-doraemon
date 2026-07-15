# 贡献指南

感谢您对 **RL-Doraemon** 麻将 AI 强化学习项目的关注！我们欢迎任何形式的贡献，无论是提交 Bug 报告、提出功能建议，还是直接贡献代码。

## 📋 目录

- [行为准则](#行为准则)
- [如何贡献](#如何贡献)
  - [提交 Issue](#提交-issue)
  - [提交 Pull Request](#提交-pull-request)
- [代码规范](#代码规范)
- [开发环境](#开发环境)
- [测试指南](#测试指南)
- [贡献类型](#贡献类型)

## 行为准则

本项目遵循 [Contributor Covenant](.github/CODE_OF_CONDUCT.md) 行为准则。参与项目即表示您同意遵守其条款。

## 如何贡献

### 提交 Issue

如果您发现了 Bug 或有新功能建议，请通过 Issue 告诉我们：

1. **Bug 报告**：请包含以下信息
   - 问题描述（清晰简洁）
   - 复现步骤
   - 预期行为
   - 实际行为
   - 环境信息（操作系统、Python 版本、PyTorch 版本等）
   - 错误日志或截图（如适用）
   - 相关的配置参数

2. **功能建议**：请包含以下信息
   - 功能描述
   - 为什么需要这个功能
   - 实现思路（可选）
   - 相关的学术论文或参考资料（如适用）

### 提交 Pull Request

1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

**PR 规范：**
- 标题清晰描述改动内容，使用中文或英文均可
- 详细说明改动的原因和内容
- 关联相关的 Issue（如 `Fixes #123`）
- 确保代码通过所有测试和代码检查
- 更新相关文档（如 README、CHANGELOG）
- 如果添加了新功能，请添加相应的测试用例

## 代码规范

项目使用以下工具进行代码规范管理：

- **Black** - 代码格式化
- **isort** - 导入排序
- **flake8** - 代码检查
- **pytest** - 单元测试

**代码规范：**
- 使用 4 空格缩进
- 变量和函数使用 snake_case
- 类名使用 PascalCase
- 常量使用 UPPER_SNAKE_CASE
- 函数和类需要 docstring 说明（使用 Google 风格）
- 类型注解（Type Hints）是推荐的
- 避免魔法数字，使用常量定义

**运行代码检查：**
```bash
# 格式化代码
black .
isort .

# 代码检查
flake8 .

# 运行测试
pytest tests/ -v
```

## 代码风格指南

### Git 提交规范

我们遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>(<scope>): <subject>

<body>

<footer>
```

**提交类型（type）：**
- `feat` - 新功能、新算法
- `fix` - Bug 修复
- `docs` - 文档更新
- `style` - 代码格式（不影响代码运行）
- `refactor` - 重构（既不是新增功能，也不是修改 bug）
- `perf` - 性能优化
- `test` - 增加测试
- `chore` - 构建过程或辅助工具的变动
- `ci` - CI/CD 配置变更
- `revert` - 回退提交

**示例：**
```
feat(dqn): add prioritized experience replay

- Implement PER with proportional prioritization
- Add beta annealing schedule
- Update training configuration
- Add unit tests for PER buffer

Closes #123
```

**提交规范：**
- 标题不超过 72 个字符
- 使用中文或英文均可，但要保持一致
- 标题使用祈使句（"添加" 而不是 "添加了"）
- 正文详细说明改动的原因和内容
- 关联相关 Issue（如 `Closes #123`、`Fixes #456`）

### 命名约定

```python
# 变量名 - snake_case，描述性命名
state_tensor = torch.zeros(batch_size, state_dim)
episode_rewards = []
max_gradient_norm = 10.0

# 函数名 - snake_case，动词开头
def compute_td_loss(states, actions, rewards, next_states, dones):
    """计算 TD 误差损失"""
    pass

def update_target_network(target_net, policy_net, tau):
    """软更新目标网络参数"""
    pass

# 类名 - PascalCase
class DQNAgent:
    """DQN 智能体"""
    pass

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    pass

class Trainer:
    """训练器"""
    pass

# 常量 - UPPER_SNAKE_CASE
LEARNING_RATE = 1e-4
GAMMA = 0.99
BATCH_SIZE = 64
REPLAY_BUFFER_SIZE = 100000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 私有变量/方法 - 下划线前缀
class DQNAgent:
    def __init__(self):
        self._policy_net = None
        self._target_net = None
    
    def _compute_loss(self, batch):
        """内部损失计算方法"""
        pass
```

#### 文件命名
```
# Python 文件 - snake_case
dqn_agent.py
replay_buffer.py
trainer.py
__init__.py

# 配置文件 - snake_case
config.yaml
hyperparams.py

# 测试文件 - test_ 前缀
test_dqn_agent.py
test_replay_buffer.py
```

### 注释规范

#### Docstring（Google 风格）
```python
def train_episode(env, agent, epsilon, max_steps=1000):
    """训练一个回合

    Args:
        env: 游戏环境实例
        agent: 强化学习智能体
        epsilon: 探索率
        max_steps: 最大步数限制

    Returns:
        包含回合信息的字典：
        - reward: 总奖励
        - steps: 步数
        - loss: 平均损失

    Raises:
        ValueError: 环境或智能体未初始化

    Example:
        >>> stats = train_episode(env, agent, epsilon=0.1)
        >>> print(f"Reward: {stats['reward']}, Steps: {stats['steps']}")
    """
    pass
```

#### 行内注释
```python
# ✅ 好的注释 - 解释为什么这样做
# 使用 n-step 返回减少方差，提升样本效率
returns = compute_n_step_returns(rewards, values, n=3)

# ✅ 好的注释 - 解释复杂的数学公式
# 计算 GAE (Generalized Advantage Estimation)
# 参考: https://arxiv.org/abs/1506.02438
advantages = compute_gae(rewards, values, gamma=0.99, lam=0.95)

# ❌ 不好的注释 - 重复代码内容
# 计算损失
loss = criterion(q_values, target_q)
```

### 导入排序规范

```python
# 1. 标准库
import os
import sys
import argparse
from collections import deque
from typing import List, Tuple, Optional

# 2. 第三方库
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 3. 本地库
from rl_doraemon.agents.dqn import DQNAgent
from rl_doraemon.buffers.replay_buffer import ReplayBuffer
from rl_doraemon.utils.logger import Logger
from rl_doraemon.config import Config
```

### 错误处理规范

```python
# ✅ 使用自定义异常
class TrainingError(Exception):
    """训练过程异常"""
    def __init__(self, message: str, episode: int = 0):
        self.episode = episode
        super().__init__(f"Episode {episode}: {message}")

# ✅ 捕获具体异常
try:
    state = env.reset()
except gym.error.ResetNeeded as e:
    logger.warning(f"环境需要重置: {e}")
    state = env.reset(force=True)
except Exception as e:
    logger.error(f"未知环境错误: {e}")
    raise TrainingError(f"环境初始化失败: {e}")

# ❌ 不要静默异常
try:
    agent.update(batch)
except:  # 太宽泛且静默
    pass
```

## 开发环境

1. **克隆仓库**
   ```bash
   git clone https://github.com/lccuhk/rl-doraemon.git
   cd rl-doraemon
   ```

2. **创建虚拟环境**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**
   ```bash
   # 运行简单测试
   python simple_test.py
   
   # 运行单元测试
   pytest tests/ -v
   ```

5. **开始开发**
   ```bash
   # 训练模型（示例）
   python examples/train_dqn.py
   
   # 评估模型
   python examples/evaluate_model.py
   ```

## 测试指南

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_environment.py -v

# 运行特定测试函数
pytest tests/test_environment.py::test_mahjong_env -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 编写测试

- 测试文件放在 `tests/` 目录下
- 测试函数以 `test_` 开头
- 使用 pytest 框架
- 为新功能添加相应的单元测试
- 确保测试覆盖关键逻辑路径

## 贡献类型

我们欢迎各种类型的贡献：

### 🐛 Bug 修复
- 修复游戏环境中的逻辑错误
- 修复训练过程中的稳定性问题
- 修复神经网络模型的实现问题

### ✨ 新功能
- 添加新的麻将规则（国标、四川麻将等）
- 实现新的强化学习算法（PPO、A2C、DQN 变体等）
- 添加新的神经网络架构
- 实现多智能体训练
- 添加模仿学习功能

### 📚 文档
- 改进 README 文档
- 添加使用教程和示例
- 补充代码注释
- 翻译文档

### 🎨 代码质量
- 重构代码以提高可读性
- 优化性能
- 添加类型注解
- 改进测试覆盖率

### 📊 研究贡献
- 提出新的训练方法
- 对比不同算法的性能
- 分享训练经验和调参技巧

## 问题？

如果您在贡献过程中遇到任何问题，欢迎通过以下方式联系我们：

- 提交 [Issue](https://github.com/lccuhk/rl-doraemon/issues)
- 查看 [README.md](README.md) 了解更多项目信息
- 查看 [CHANGELOG.md](CHANGELOG.md) 了解版本历史

再次感谢您的贡献！🎉
