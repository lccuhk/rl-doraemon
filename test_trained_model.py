#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.mahjong_env import MahjongEnv, GamePhase
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.utils.tile_utils import TileUtils

def test_trained_model():
    print("="*60)
    print("🀄 测试训练好的DQN模型")
    print("="*60)
    
    checkpoint_path = "./checkpoints/checkpoint_199_20260512_120523_agent0.pt"
    
    print(f"\n加载模型: {checkpoint_path}")
    
    dqn_agent = DQNAgent(
        name="Trained_DQN",
        input_shape=(6, 34, 4),
        num_actions=38,
        hidden_size=256,
        learning_rate=0.001,
        seed=42
    )
    
    dqn_agent.load(checkpoint_path)
    dqn_agent.set_training(False)
    
    print(f"\n模型信息:")
    print(f"   - 设备: {dqn_agent.device}")
    print(f"   - 训练步数: {dqn_agent.steps_done}")
    print(f"   - Epsilon: {dqn_agent.get_epsilon():.4f}")
    
    print("\n" + "="*60)
    print("开始测试游戏")
    print("="*60)
    
    env = MahjongEnv(seed=123)
    
    ai_opponent1 = RandomAgent("AI_南", seed=43)
    ai_opponent2 = RandomAgent("AI_西", seed=44)
    ai_opponent3 = RandomAgent("AI_北", seed=45)
    
    agents = [dqn_agent, ai_opponent1, ai_opponent2, ai_opponent3]
    
    obs = env.reset()
    done = False
    step_count = 0
    max_steps = 200
    
    print(f"\n初始状态:")
    print(f"   - 玩家0 (DQN) 手牌数: {len(env.players[0]['hand'])}")
    print(f"   - 牌山剩余: {len(env.wall)}")
    print(f"   - 宝牌指示牌: {TileUtils.get_tile_symbol(env.dora_indicators[0])} {TileUtils.get_tile_name(env.dora_indicators[0])}")
    
    print(f"\n玩家0 (DQN) 初始手牌:")
    hand = env.players[0]['hand']
    symbols = [TileUtils.get_tile_symbol(t) for t in hand]
    names = [TileUtils.get_tile_name(t) for t in hand]
    print(f"   符号: {' '.join(symbols)}")
    print(f"   名称: {' '.join(names)}")
    
    print("\n" + "="*60)
    print("游戏进行中...")
    print("="*60)
    
    dqn_actions = []
    dqn_discards = []
    
    while not done and step_count < max_steps:
        current_player = obs['current_player']
        legal_actions = obs['legal_actions']
        
        if current_player == 0 and step_count < 30:
            print(f"\n--- 第 {step_count + 1} 步 (玩家0: DQN) ---")
            print(f"当前阶段: {GamePhase(obs['phase']).name}")
            
            hand = obs['hand']
            symbols = [TileUtils.get_tile_symbol(t) for t in hand]
            print(f"手牌: {' '.join(symbols)}")
            print(f"合法动作: {legal_actions}")
        
        action = agents[current_player].select_action(obs, legal_actions)
        
        if current_player == 0:
            dqn_actions.append((step_count, action, GamePhase(obs['phase']).name))
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        if current_player == 0 and 'discarded' in info:
            tile = info['discarded']
            dqn_discards.append((step_count, tile))
            if step_count <= 30:
                print(f"DQN 舍牌: {TileUtils.get_tile_symbol(tile)} {TileUtils.get_tile_name(tile)}")
    
    print("\n" + "="*60)
    print("游戏结束！")
    print("="*60)
    
    if env.winner >= 0:
        winner_name = agents[env.winner].name
        winner_type = "DQN模型" if env.winner == 0 else "随机AI"
        print(f"\n🏆 获胜者: 玩家{env.winner} ({winner_name}) - {winner_type}")
    else:
        print("\n🤝 流局（牌山用完）")
    
    print(f"\n总步数: {step_count}")
    
    print("\n最终分数:")
    for i, player in enumerate(env.players):
        agent_type = "DQN模型" if i == 0 else "随机AI"
        status = []
        if i == env.dealer:
            status.append("庄家")
        if env.is_riichi[i]:
            status.append("立直")
        status_str = f" ({', '.join(status)})" if status else ""
        print(f"   玩家{i} ({agents[i].name} - {agent_type}): {player['score']}点{status_str}")
    
    print(f"\nDQN模型动作统计:")
    print(f"   - 总动作数: {len(dqn_actions)}")
    
    phase_counts = {}
    for _, _, phase in dqn_actions:
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
    
    for phase, count in phase_counts.items():
        print(f"   - {phase}: {count}次")
    
    if dqn_discards:
        print(f"\nDQN模型舍牌记录 (前10张):")
        for i, (step, tile) in enumerate(dqn_discards[:10]):
            print(f"   第{step}步: {TileUtils.get_tile_symbol(tile)} {TileUtils.get_tile_name(tile)}")
    
    print("\n舍牌记录:")
    for i in range(4):
        if env.discards[i]:
            symbols = [TileUtils.get_tile_symbol(t) for t in env.discards[i]]
            agent_type = "DQN" if i == 0 else "AI"
            print(f"   玩家{i} ({agent_type}): {' '.join(symbols)}")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    
    return env.winner == 0

if __name__ == "__main__":
    success = test_trained_model()
    
    if success:
        print("\n🎉 DQN模型获胜！")
    else:
        print("\n📊 DQN模型需要更多训练...")
