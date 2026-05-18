import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent

def main():
    print("="*60)
    print("麻将AI - 详细模型分析")
    print("="*60)
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong_v2",
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
    
    model_path = "./checkpoints_v2/final_model_20260512_140810_agent0.pt"
    dqn_agent.load(model_path)
    dqn_agent.set_training(False)
    
    print(f"已加载模型: {model_path}")
    
    opponent = RandomAgent("Opponent", seed=43)
    agents = [dqn_agent, opponent, opponent, opponent]
    
    num_games = 50
    print(f"\n开始分析 ({num_games}局)...")
    
    stats = {
        'wins': [0] * 4,
        'draws': 0,
        'tsumo_wins': [0] * 4,
        'ron_wins': [0] * 4,
        'tenpai_count': [0] * 4,
        'total_turns': 0,
        'games_with_win': 0
    }
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        game_turns = 0
        
        for agent in agents:
            agent.reset()
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            next_obs, reward, done, info = env.step(action)
            game_turns += 1
            obs = next_obs
        
        stats['total_turns'] += game_turns
        
        if env.winner >= 0:
            stats['wins'][env.winner] += 1
            stats['games_with_win'] += 1
            
            if info.get('win_type') == 'tsumo':
                stats['tsumo_wins'][env.winner] += 1
            elif info.get('win_type') == 'ron':
                stats['ron_wins'][env.winner] += 1
        else:
            stats['draws'] += 1
    
    print("\n" + "="*60)
    print("详细分析结果")
    print("="*60)
    print(f"总局数: {num_games}")
    print(f"有和牌的局数: {stats['games_with_win']} ({stats['games_with_win']/num_games*100:.1f}%)")
    print(f"流局数: {stats['draws']} ({stats['draws']/num_games*100:.1f}%)")
    print(f"平均每局回合数: {stats['total_turns']/num_games:.1f}")
    
    print("\n各玩家统计:")
    for i in range(4):
        agent_type = "DQN" if i == 0 else "Random"
        win_rate = stats['wins'][i] / num_games * 100
        print(f"\n  玩家{i} ({agent_type}):")
        print(f"    总胜场: {stats['wins'][i]} ({win_rate:.1f}%)")
        print(f"    自摸胜场: {stats['tsumo_wins'][i]}")
        print(f"    荣和胜场: {stats['ron_wins'][i]}")
    
    if stats['games_with_win'] > 0:
        print(f"\nDQN vs 随机对手胜率 (仅考虑有和牌的局):")
        dqn_win_rate = stats['wins'][0] / stats['games_with_win'] * 100
        print(f"  DQN: {dqn_win_rate:.1f}%")
        opponent_avg = sum(stats['wins'][1:]) / 3 / stats['games_with_win'] * 100
        print(f"  随机对手平均: {opponent_avg:.1f}%")

if __name__ == "__main__":
    main()
