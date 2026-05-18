import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.dqn_agent import DQNAgent
from src.environment.rules import MahjongRules

def format_tile(tile_id):
    if tile_id < 0:
        return "?"
    suits = ['万', '筒', '条']
    honors = ['东', '南', '西', '北', '白', '发', '中']
    
    if tile_id < 27:
        suit = tile_id // 9
        value = tile_id % 9 + 1
        return f"{value}{suits[suit]}"
    else:
        return honors[tile_id - 27]

def format_hand(hand):
    return ' '.join([format_tile(t) for t in sorted(hand)])

def analyze_hand(hand, melds=None):
    analysis = {
        'hand_count': len(hand),
        'meld_count': len(melds) if melds else 0,
        'shanten': None,
        'is_tenpai': False,
        'can_win': False,
        'win_type': None
    }
    
    try:
        analysis['shanten'] = MahjongRules.calculate_shanten(hand, melds if melds else [])
        analysis['is_tenpai'] = analysis['shanten'] == 0
    except:
        pass
    
    if len(hand) == 14:
        try:
            win_result = MahjongRules.check_win(hand, melds if melds else [])
            analysis['can_win'] = win_result['can_win']
            analysis['win_type'] = win_result.get('yaku', [])
        except:
            pass
    
    return analysis

def main():
    print("="*70)
    print("麻将AI - 模型和牌能力测试")
    print("="*70)
    
    model_path = "./checkpoints_realtime/final_model_20260513_212614_agent0.pt"
    
    if not os.path.exists(model_path):
        print(f"\n❌ 模型文件不存在: {model_path}")
        print("请先运行训练脚本 train_with_monitor.py")
        return
    
    print(f"\n📦 加载模型: {model_path}")
    
    env = MahjongEnv(seed=42, use_intermediate_rewards=True)
    
    dqn_agent = DQNAgent(
        name="DQN_Mahjong",
        input_shape=(6, 34, 4),
        num_actions=38,
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
    
    dqn_agent.load(model_path)
    dqn_agent.set_training(False)
    
    print("✅ 模型加载成功！")
    
    opponent = RandomAgent("Opponent", seed=43)
    agents = [dqn_agent, opponent, opponent, opponent]
    
    num_games = 50
    print(f"\n🎮 开始测试 ({num_games}局)...")
    print("-"*70)
    
    stats = {
        'total_games': num_games,
        'wins': [0] * 4,
        'draws': 0,
        'tsumo_wins': [0] * 4,
        'ron_wins': [0] * 4,
        'total_turns': 0,
        'games_with_win': 0,
        'dqn_tenpai_count': 0,
        'dqn_meld_count': 0,
        'dqn_riichi_count': 0,
        'total_rewards': [0.0] * 4,
        'reward_wins': [0] * 4,
        'hand_analyses': []
    }
    
    for game in range(num_games):
        obs = env.reset()
        done = False
        game_turns = 0
        game_rewards = [0.0] * 4
        game_info = {
            'game': game + 1,
            'dqn_hands': [],
            'dqn_actions': [],
            'winner': -1,
            'win_type': None
        }
        
        for agent in agents:
            agent.reset()
        
        while not done:
            current_player = obs['current_player']
            legal_actions = obs['legal_actions']
            
            agent = agents[current_player]
            action = agent.select_action(obs, legal_actions)
            
            if current_player == 0:
                hand = obs.get('observation', {}).get('hand', [])
                if len(hand) > 0:
                    analysis = analyze_hand(hand)
                    game_info['dqn_hands'].append({
                        'turn': game_turns,
                        'hand': format_hand(hand),
                        'shanten': analysis['shanten'],
                        'is_tenpai': analysis['is_tenpai'],
                        'can_win': analysis['can_win']
                    })
                    
                    if analysis['is_tenpai']:
                        stats['dqn_tenpai_count'] += 1
            
            next_obs, reward, done, info = env.step(action)
            game_rewards[current_player] += reward
            game_turns += 1
            obs = next_obs
        
        stats['total_turns'] += game_turns
        
        for i, r in enumerate(game_rewards):
            stats['total_rewards'][i] += r
            if r > 0:
                stats['reward_wins'][i] += 1
        
        if env.winner >= 0:
            stats['wins'][env.winner] += 1
            stats['games_with_win'] += 1
            game_info['winner'] = env.winner
            
            if info.get('win_type') == 'tsumo':
                stats['tsumo_wins'][env.winner] += 1
                game_info['win_type'] = 'tsumo'
            elif info.get('win_type') == 'ron':
                stats['ron_wins'][env.winner] += 1
                game_info['win_type'] = 'ron'
        else:
            stats['draws'] += 1
        
        stats['hand_analyses'].append(game_info)
        
        if (game + 1) % 10 == 0:
            print(f"  已完成 {game + 1}/{num_games} 局...")
    
    print("\n" + "="*70)
    print("📊 测试结果统计")
    print("="*70)
    
    print(f"\n📈 基本统计:")
    print(f"  总局数: {stats['total_games']}")
    print(f"  有和牌的局数: {stats['games_with_win']} ({stats['games_with_win']/stats['total_games']*100:.1f}%)")
    print(f"  流局数: {stats['draws']} ({stats['draws']/stats['total_games']*100:.1f}%)")
    print(f"  平均每局回合数: {stats['total_turns']/stats['total_games']:.1f}")
    
    print(f"\n🎯 基于奖励的胜率:")
    for i in range(4):
        agent_type = "DQN" if i == 0 else "Random"
        win_rate = stats['reward_wins'][i] / stats['total_games'] * 100
        avg_reward = stats['total_rewards'][i] / stats['total_games']
        print(f"  玩家{i} ({agent_type}): 胜率={win_rate:.1f}%, 平均奖励={avg_reward:.3f}")
    
    print(f"\n🏆 基于实际和牌的统计:")
    for i in range(4):
        agent_type = "DQN" if i == 0 else "Random"
        win_rate = stats['wins'][i] / stats['total_games'] * 100
        print(f"\n  玩家{i} ({agent_type}):")
        print(f"    总胜场: {stats['wins'][i]} ({win_rate:.1f}%)")
        print(f"    自摸胜场: {stats['tsumo_wins'][i]}")
        print(f"    荣和胜场: {stats['ron_wins'][i]}")
    
    print(f"\n📊 DQN 详细分析:")
    print(f"  听牌次数: {stats['dqn_tenpai_count']}")
    print(f"  平均每局听牌次数: {stats['dqn_tenpai_count']/stats['total_games']:.2f}")
    
    if stats['games_with_win'] > 0:
        print(f"\n⚖️ DQN vs 随机对手胜率 (仅考虑有和牌的局):")
        dqn_win_rate = stats['wins'][0] / stats['games_with_win'] * 100
        print(f"  DQN: {dqn_win_rate:.1f}%")
        opponent_avg = sum(stats['wins'][1:]) / 3 / stats['games_with_win'] * 100
        print(f"  随机对手平均: {opponent_avg:.1f}%")
    
    print("\n" + "="*70)
    print("🔍 详细对局分析 (前5局)")
    print("="*70)
    
    for i, game_info in enumerate(stats['hand_analyses'][:5]):
        print(f"\n--- 第 {game_info['game']} 局 ---")
        
        if game_info['winner'] >= 0:
            winner_type = "DQN" if game_info['winner'] == 0 else "Random"
            win_type_str = "自摸" if game_info['win_type'] == 'tsumo' else "荣和" if game_info['win_type'] == 'ron' else ""
            print(f"  获胜者: 玩家{game_info['winner']} ({winner_type}) {win_type_str}")
        else:
            print(f"  结果: 流局")
        
        print(f"  DQN 手牌变化:")
        for hand_info in game_info['dqn_hands'][-3:]:
            tenpai_str = "✅ 听牌" if hand_info['is_tenpai'] else "⏳ 向听"
            shanten_str = f"向听{hand_info['shanten']}" if hand_info['shanten'] is not None else "未知"
            print(f"    回合{hand_info['turn']}: {hand_info['hand']} [{tenpai_str}, {shanten_str}]")
    
    print("\n" + "="*70)
    print("📝 总结")
    print("="*70)
    
    if stats['games_with_win'] == 0:
        print("\n⚠️ 所有对局都是流局，没有实际和牌发生。")
        print("\n可能的原因:")
        print("  1. 麻将和牌需要特定的牌型组合（4面子+1雀头）")
        print("  2. 需要有至少一番的役（如立直、断幺九等）")
        print("  3. 模型可能更关注中间奖励而非实际和牌")
        print("\n建议:")
        print("  1. 增加实际和牌的奖励权重")
        print("  2. 增加训练回合数")
        print("  3. 考虑使用自我对弈（self-play）")
    else:
        dqn_wins = stats['wins'][0]
        print(f"\n✅ 有 {stats['games_with_win']} 局实现了实际和牌！")
        print(f"   DQN 获胜 {dqn_wins} 局")

if __name__ == "__main__":
    main()
