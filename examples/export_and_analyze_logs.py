import sys
import os
import json
import glob
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_training_data(log_dir="./logs_high_reward"):
    print("="*70)
    print("加载训练数据")
    print("="*70)
    
    status_file = os.path.join(log_dir, 'realtime_status.json')
    history_files = sorted(glob.glob(os.path.join(log_dir, 'training_history_*.json')))
    
    data = {
        'status': None,
        'history_files': history_files,
        'all_rewards': [],
        'all_episodes': [],
        'evaluations': []
    }
    
    if os.path.exists(status_file):
        with open(status_file, 'r', encoding='utf-8') as f:
            data['status'] = json.load(f)
        print(f"当前状态: {data['status']['current_episode']}/{data['status']['total_episodes']} 回合")
        print(f"进度: {data['status']['progress']:.1f}%")
        print(f"平均奖励: {data['status']['avg_reward']:.4f}")
        print(f"已用时间: {int(data['status']['elapsed_seconds']/60)}分钟")
        
        if 'evaluation_results' in data['status'] and data['status']['evaluation_results']:
            data['evaluations'] = data['status']['evaluation_results']
            print(f"评估次数: {len(data['evaluations'])}")
    
    print(f"历史文件数: {len(history_files)}")
    
    return data

def analyze_win_failures(data, output_dir="./logs_high_reward/analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    status = data['status']
    evaluations = data['evaluations']
    recent_rewards = status.get('recent_rewards', [])
    
    print("\n" + "="*70)
    print("和牌失败原因分析")
    print("="*70)
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'current_episode': status['current_episode'],
        'total_episodes': status['total_episodes'],
        'progress': status['progress'],
        'avg_reward': status['avg_reward'],
        'elapsed_minutes': int(status['elapsed_seconds']/60),
        'recent_rewards': recent_rewards,
        'evaluations': evaluations,
        'analysis': {}
    }
    
    print(f"\n【训练状态】")
    print(f"  当前回合: {status['current_episode']}/{status['total_episodes']}")
    print(f"  进度: {status['progress']:.1f}%")
    print(f"  平均奖励: {status['avg_reward']:.4f}")
    print(f"  已用时间: {int(status['elapsed_seconds']/60)}分钟")
    
    print(f"\n【最近奖励趋势】")
    if recent_rewards:
        max_reward = max(recent_rewards)
        min_reward = min(recent_rewards)
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        print(f"  最近{len(recent_rewards)}回合:")
        print(f"    最大奖励: {max_reward:.4f}")
        print(f"    最小奖励: {min_reward:.4f}")
        print(f"    平均奖励: {avg_reward:.4f}")
        
        high_rewards = [(i, r) for i, r in enumerate(recent_rewards) if r > 3.0]
        if high_rewards:
            print(f"    高奖励回合(>3.0): {len(high_rewards)}个")
            for i, r in high_rewards:
                print(f"      回合{i+1}: {r:.4f}")
        
        analysis['analysis']['recent_rewards'] = {
            'max': max_reward,
            'min': min_reward,
            'avg': avg_reward,
            'high_rewards_count': len(high_rewards)
        }
    
    print(f"\n【评估结果分析】")
    if evaluations:
        print(f"  评估次数: {len(evaluations)}")
        
        dqn_rewards = [e['avg_rewards'][0] for e in evaluations]
        actual_win_rates = [e['actual_win_rates'][0] for e in evaluations]
        draw_rates = [e['draw_rate'] for e in evaluations]
        
        print(f"\n  DQN奖励趋势:")
        for i, (r, wr, dr) in enumerate(zip(dqn_rewards, actual_win_rates, draw_rates)):
            eval_episode = evaluations[i]['episode'] + 1
            print(f"    评估{i+1} (回合{eval_episode}): 奖励={r:.4f}, 和牌率={wr*100:.1f}%, 流局率={dr*100:.1f}%")
        
        if len(dqn_rewards) > 1:
            trend = dqn_rewards[-1] - dqn_rewards[0]
            print(f"\n  奖励趋势: {'上升' if trend > 0 else '下降'} ({trend:+.4f})")
        
        analysis['analysis']['evaluations'] = {
            'count': len(evaluations),
            'dqn_rewards': dqn_rewards,
            'actual_win_rates': actual_win_rates,
            'draw_rates': draw_rates,
            'reward_trend': dqn_rewards[-1] - dqn_rewards[0] if len(dqn_rewards) > 1 else 0
        }
    
    print(f"\n【和牌失败原因分析】")
    print(f"  1. 训练时间不足: {status['progress']:.1f}% 完成，需要更多回合")
    print(f"  2. 和牌是稀疏奖励: 即使有优先和牌逻辑，也需要机会出现")
    print(f"  3. epsilon探索: 初期epsilon高，随机动作可能错过和牌机会")
    print(f"  4. 对手策略: 随机对手不会主动送和牌")
    
    print(f"\n【建议】")
    print(f"  1. 继续训练到3000回合")
    print(f"  2. 观察奖励峰值是否持续出现")
    print(f"  3. 检查日志中是否有[和牌机会]记录")
    
    analysis['analysis']['possible_causes'] = [
        "训练时间不足",
        "和牌是稀疏奖励",
        "epsilon探索阶段",
        "随机对手策略"
    ]
    
    analysis['analysis']['recommendations'] = [
        "继续训练到3000回合",
        "观察奖励峰值是否持续出现",
        "检查日志中是否有[和牌机会]记录"
    ]
    
    output_file = os.path.join(output_dir, f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    print(f"\n分析结果已保存到: {output_file}")
    
    return analysis

def generate_html_report(data, analysis, output_dir="./logs_high_reward/analysis"):
    os.makedirs(output_dir, exist_ok=True)
    
    status = data['status']
    evaluations = data['evaluations']
    recent_rewards = status.get('recent_rewards', [])
    
    episodes = []
    dqn_rewards = []
    opp1_rewards = []
    opp2_rewards = []
    opp3_rewards = []
    actual_win_rates = []
    draw_rates = []
    
    for eval_result in evaluations:
        episodes.append(eval_result['episode'] + 1)
        dqn_rewards.append(eval_result['avg_rewards'][0])
        opp1_rewards.append(eval_result['avg_rewards'][1])
        opp2_rewards.append(eval_result['avg_rewards'][2])
        opp3_rewards.append(eval_result['avg_rewards'][3])
        actual_win_rates.append(eval_result['actual_win_rates'][0] * 100)
        draw_rates.append(eval_result['draw_rate'] * 100)
    
    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>训练日志分析报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        h2 {{
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .summary-item {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-item .label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}
        .summary-item .value {{
            color: #333;
            font-size: 24px;
            font-weight: bold;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            color: #333;
            margin-bottom: 15px;
            font-size: 18px;
            font-weight: bold;
        }}
        svg {{
            width: 100%;
            height: 300px;
        }}
        .axis path, .axis line {{
            stroke: #ccc;
        }}
        .axis text {{
            font-size: 12px;
            fill: #666;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-top: 10px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 14px;
        }}
        .legend-color {{
            width: 20px;
            height: 4px;
            border-radius: 2px;
        }}
        .analysis-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .analysis-item {{
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .analysis-item .title {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .analysis-item .content {{
            color: #666;
            font-size: 14px;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 12px;
            margin-top: 20px;
        }}
        .highlight {{
            background: #fff3cd;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
            margin-bottom: 15px;
        }}
    </style>
</head>
<body>
    <h1>训练日志分析报告</h1>
    
    <div class="summary">
        <h2 style="margin-top: 0;">训练摘要</h2>
        <div class="summary-grid">
            <div class="summary-item">
                <div class="label">当前回合</div>
                <div class="value">{status['current_episode']}/{status['total_episodes']}</div>
            </div>
            <div class="summary-item">
                <div class="label">训练进度</div>
                <div class="value">{status['progress']:.1f}%</div>
            </div>
            <div class="summary-item">
                <div class="label">平均奖励</div>
                <div class="value">{status['avg_reward']:.4f}</div>
            </div>
            <div class="summary-item">
                <div class="label">已用时间</div>
                <div class="value">{int(status['elapsed_seconds']/60)}分钟</div>
            </div>
        </div>
    </div>
"""
    
    if recent_rewards:
        max_reward = max(recent_rewards)
        min_reward = min(recent_rewards)
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        high_rewards = [(i, r) for i, r in enumerate(recent_rewards) if r > 3.0]
        
        html_content += f"""
    <div class="highlight">
        <strong>最近奖励分析:</strong>
        <br>最大奖励: {max_reward:.4f} | 最小奖励: {min_reward:.4f} | 平均奖励: {avg_reward:.4f}
        <br>高奖励回合(>3.0): {len(high_rewards)}个
        {' - 这表明模型可能开始学习有效的策略!' if len(high_rewards) > 0 else ''}
    </div>
    
    <div class="chart-container">
        <div class="chart-title">最近奖励趋势（最近{len(recent_rewards)}回合）</div>
        <svg id="recentChart" viewBox="0 0 800 300">
            <g class="axis">
                <line x1="50" y1="250" x2="780" y2="250" stroke="#ccc"/>
                <line x1="50" y1="30" x2="50" y2="250" stroke="#ccc"/>
            </g>
"""
        range_reward = max_reward - min_reward if max_reward != min_reward else 1
        
        points = []
        for i, reward in enumerate(recent_rewards):
            x = 50 + (i / (len(recent_rewards) - 1)) * 730 if len(recent_rewards) > 1 else 415
            y = 250 - ((reward - min_reward) / range_reward) * 200
            points.append(f"{x},{y}")
        
        path_d = f"M {points[0]}"
        for p in points[1:]:
            path_d += f" L {p}"
        
        html_content += f"""
            <path d="{path_d}" fill="none" stroke="#4CAF50" stroke-width="2"/>
"""
        
        for i, (x, y, reward) in enumerate(zip([50 + (i / (len(recent_rewards) - 1)) * 730 if len(recent_rewards) > 1 else 415 for i in range(len(recent_rewards))], 
                                          [250 - ((r - min_reward) / range_reward) * 200 for r in recent_rewards],
                                          recent_rewards)):
            if reward > 3.0:
                html_content += f'<circle cx="{x}" cy="{y}" r="5" fill="#FF5722"/>'
            else:
                html_content += f'<circle cx="{x}" cy="{y}" r="3" fill="#4CAF50"/>'
        
        html_content += f"""
            <text x="30" y="35" fill="#666" font-size="12">{max_reward:.2f}</text>
            <text x="30" y="255" fill="#666" font-size="12">{min_reward:.2f}</text>
        </svg>
        <div class="legend">
            <div class="legend-item"><span class="legend-color" style="background: #4CAF50;"></span>正常奖励</div>
            <div class="legend-item"><span class="legend-color" style="background: #FF5722;"></span>高奖励(>3.0)</div>
        </div>
    </div>
"""
    
    if evaluations:
        html_content += f"""
    <div class="chart-container">
        <div class="chart-title">评估奖励对比（每50回合评估）</div>
        <svg id="rewardChart" viewBox="0 0 800 300">
            <g class="axis">
                <line x1="50" y1="250" x2="780" y2="250" stroke="#ccc"/>
                <line x1="50" y1="30" x2="50" y2="250" stroke="#ccc"/>
            </g>
"""
        
        all_rewards = dqn_rewards + opp1_rewards + opp2_rewards + opp3_rewards
        max_r = max(all_rewards) if all_rewards else 1
        min_r = min(all_rewards) if all_rewards else 0
        range_r = max_r - min_r if max_r != min_r else 1
        
        colors = ['#2196F3', '#FF9800', '#9C27B0', '#F44336']
        labels = ['DQN Agent', '对手1', '对手2', '对手3']
        reward_lists = [dqn_rewards, opp1_rewards, opp2_rewards, opp3_rewards]
        
        for color, rewards in zip(colors, reward_lists):
            if rewards:
                points = []
                for i, r in enumerate(rewards):
                    x = 50 + (i / (len(rewards) - 1)) * 730 if len(rewards) > 1 else 415
                    y = 250 - ((r - min_r) / range_r) * 200
                    points.append(f"{x},{y}")
                
                if points:
                    path_d = f"M {points[0]}"
                    for p in points[1:]:
                        path_d += f" L {p}"
                    html_content += f'<path d="{path_d}" fill="none" stroke="{color}" stroke-width="2"/>'
        
        html_content += f"""
            <text x="30" y="35" fill="#666" font-size="12">{max_r:.2f}</text>
            <text x="30" y="255" fill="#666" font-size="12">{min_r:.2f}</text>
        </svg>
        <div class="legend">
"""
        for color, label in zip(colors, labels):
            html_content += f'<div class="legend-item"><span class="legend-color" style="background: {color};"></span>{label}</div>'
        
        html_content += """
        </div>
    </div>
    
    <div class="chart-container">
        <div class="chart-title">实际和牌率与流局率</div>
        <svg id="winRateChart" viewBox="0 0 800 300">
            <g class="axis">
                <line x1="50" y1="250" x2="780" y2="250" stroke="#ccc"/>
                <line x1="50" y1="30" x2="50" y2="250" stroke="#ccc"/>
            </g>
"""
        
        if actual_win_rates:
            points_win = []
            points_draw = []
            for i, (wr, dr) in enumerate(zip(actual_win_rates, draw_rates)):
                x = 50 + (i / (len(actual_win_rates) - 1)) * 730 if len(actual_win_rates) > 1 else 415
                y_win = 250 - (wr / 100) * 200
                y_draw = 250 - (dr / 100) * 200
                points_win.append(f"{x},{y_win}")
                points_draw.append(f"{x},{y_draw}")
            
            if points_win:
                path_win = f"M {points_win[0]}"
                for p in points_win[1:]:
                    path_win += f" L {p}"
                html_content += f'<path d="{path_win}" fill="none" stroke="#4CAF50" stroke-width="2"/>'
            
            if points_draw:
                path_draw = f"M {points_draw[0]}"
                for p in points_draw[1:]:
                    path_draw += f" L {p}"
                html_content += f'<path d="{path_draw}" fill="none" stroke="#FF5722" stroke-width="2"/>'
        
        html_content += """
            <text x="30" y="35" fill="#666" font-size="12">100%</text>
            <text x="30" y="255" fill="#666" font-size="12">0%</text>
        </svg>
        <div class="legend">
            <div class="legend-item"><span class="legend-color" style="background: #4CAF50;"></span>DQN和牌率</div>
            <div class="legend-item"><span class="legend-color" style="background: #FF5722;"></span>流局率</div>
        </div>
    </div>
"""
    
    html_content += """
    <div class="analysis-section">
        <h2 style="margin-top: 0;">和牌失败原因分析</h2>
        
        <div class="analysis-item">
            <div class="title">1. 训练时间不足</div>
            <div class="content">当前仅完成 %.1f%% 的训练，DQN需要大量样本才能学习到和牌策略。建议继续训练到3000回合。</div>
        </div>
        
        <div class="analysis-item">
            <div class="title">2. 和牌是稀疏奖励</div>
            <div class="content">麻将和牌本身就是低概率事件，即使有优先和牌逻辑，也需要和牌机会出现才能学习。</div>
        </div>
        
        <div class="analysis-item">
            <div class="title">3. epsilon探索阶段</div>
            <div class="content">训练初期epsilon较高，随机动作可能错过和牌机会。随着训练进行，epsilon会逐渐降低。</div>
        </div>
        
        <div class="analysis-item">
            <div class="title">4. 随机对手策略</div>
            <div class="content">对手是随机策略，不会主动送和牌，荣和机会较少。主要依赖自摸。</div>
        </div>
    </div>
    
    <div class="analysis-section">
        <h2 style="margin-top: 0;">建议</h2>
        
        <div class="analysis-item">
            <div class="title">1. 继续训练</div>
            <div class="content">继续训练到3000回合，观察模型是否能学习到和牌策略。</div>
        </div>
        
        <div class="analysis-item">
            <div class="title">2. 监控奖励峰值</div>
            <div class="content">最近出现了高奖励回合(>3.0)，这可能是模型开始学习的信号。继续观察是否持续出现。</div>
        </div>
        
        <div class="analysis-item">
            <div class="title">3. 检查日志</div>
            <div class="content">训练日志中会记录[和牌机会]和[优先和牌]等信息，关注这些日志可以了解和牌机会是否出现。</div>
        </div>
    </div>
""" % (status['progress'])
    
    html_content += f"""
    <div class="timestamp">
        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
    
    output_file = os.path.join(output_dir, f'analysis_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nHTML报告已保存到: {output_file}")
    
    return output_file

def main():
    data = load_training_data()
    
    if data['status']:
        analysis = analyze_win_failures(data)
        html_file = generate_html_report(data, analysis)
        
        print(f"\n" + "="*70)
        print("导出完成!")
        print("="*70)
        print(f"请在浏览器中打开以下文件查看详细分析报告:")
        print(f"file://{os.path.abspath(html_file)}")
    else:
        print("错误: 无法加载训练状态文件")

if __name__ == "__main__":
    main()
