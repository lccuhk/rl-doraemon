import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv, ActionType, GamePhase
from src.utils.tile_utils import TileUtils
from src.agents.dqn_agent import DQNAgent

def main():
    print("="*70)
    print("手动测试和牌日志功能")
    print("="*70)
    
    print("\n【测试说明】")
    print("  手动构造一个和牌牌型，测试日志输出")
    print("="*70 + "\n")
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    
    dqn_agent = DQNAgent(
        name="DQN_Test",
        input_shape=(6, 34, 4),
        num_actions=41,
        hidden_size=256,
        learning_rate=0.001,
        gamma=0.99,
        epsilon_start=0.0,
        epsilon_end=0.0,
        epsilon_decay=1.0,
        batch_size=64,
        target_update_freq=1000,
        buffer_capacity=100000,
        seed=42
    )
    
    obs = env.reset()
    
    print("\n【构造和牌牌型】")
    print("牌型: 222万 345万 678万 222筒 33筒 (断幺九 + 一杯口)")
    
    win_hand = []
    for _ in range(3):
        win_hand.append(TileUtils.create_tile('wan', 2))
    for v in [3, 4, 5]:
        win_hand.append(TileUtils.create_tile('wan', v))
    for v in [6, 7, 8]:
        win_hand.append(TileUtils.create_tile('wan', v))
    for _ in range(3):
        win_hand.append(TileUtils.create_tile('tong', 2))
    for _ in range(2):
        win_hand.append(TileUtils.create_tile('tong', 3))
    
    print(f"手牌数量: {len(win_hand)}")
    print(f"手牌: {[TileUtils.get_tile_symbol(t) for t in win_hand]}")
    
    env.players[0]['hand'] = win_hand
    env.current_player = 0
    env.phase = GamePhase.DRAWING
    
    obs = env._get_observation(0)
    legal_actions = obs['legal_actions']
    
    print(f"\n【当前状态】")
    print(f"  玩家: 0")
    print(f"  阶段: {env.phase.name}")
    print(f"  手牌数量: {len(env.players[0]['hand'])}")
    print(f"  合法动作: {legal_actions}")
    print(f"  TSUMO动作值: {ActionType.TSUMO.value}")
    print(f"  TSUMO在合法动作中: {ActionType.TSUMO.value in legal_actions}")
    
    if ActionType.TSUMO.value in legal_actions:
        print(f"\n【测试DQN Agent决策】")
        dqn_agent.set_training(False)
        action = dqn_agent.select_action(obs, legal_actions)
        print(f"  选择动作: {action}")
        print(f"  是否选择自摸: {action == ActionType.TSUMO.value}")
        
        if action == ActionType.TSUMO.value:
            print(f"\n【执行自摸动作】")
            obs, reward, done, info = env.step(action)
            print(f"  奖励: {reward}")
            print(f"  结束: {done}")
            print(f"  赢家: {env.winner}")
            print(f"  和牌类型: {info.get('win_type', 'unknown')}")
            if 'win_info' in info:
                print(f"  和牌信息: {info['win_info']}")
    else:
        print(f"\n【问题】TSUMO不在合法动作中，可能是和牌检测有问题")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
