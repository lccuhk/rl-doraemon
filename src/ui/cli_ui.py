import sys
from typing import List, Dict, Optional
from ..environment.mahjong_env import MahjongEnv
from ..agents.base_agent import BaseAgent
from ..agents.random_agent import RandomAgent
from ..agents.human_agent import HumanAgent
from ..agents.dqn_agent import DQNAgent
from ..training.trainer import Trainer
from ..utils.tile_utils import TileUtils

class CLIUI:
    def __init__(self):
        self.env = None
        self.agents = []
        self.trainer = None
    
    def show_menu(self):
        print("\n" + "="*60)
        print("🀄 麻将AI - 深度强化学习麻将项目")
        print("="*60)
        print("\n请选择模式:")
        print("1. 人机对战 (Human vs AI)")
        print("2. AI对战 (AI vs AI)")
        print("3. 训练AI模型")
        print("4. 评估AI性能")
        print("5. 查看规则说明")
        print("0. 退出")
        print("="*60)
    
    def run(self):
        while True:
            self.show_menu()
            choice = input("\n请输入选项 (0-5): ").strip()
            
            if choice == '1':
                self.play_human_vs_ai()
            elif choice == '2':
                self.play_ai_vs_ai()
            elif choice == '3':
                self.train_ai()
            elif choice == '4':
                self.evaluate_ai()
            elif choice == '5':
                self.show_rules()
            elif choice == '0':
                print("感谢使用！再见！")
                sys.exit(0)
            else:
                print("无效选项，请重新输入。")
    
    def play_human_vs_ai(self):
        print("\n" + "="*60)
        print("人机对战模式")
        print("="*60)
        
        self.env = MahjongEnv(seed=None)
        
        human_agent = HumanAgent("Human")
        ai_agent1 = RandomAgent("AI_1", seed=42)
        ai_agent2 = RandomAgent("AI_2", seed=43)
        ai_agent3 = RandomAgent("AI_3", seed=44)
        
        self.agents = [human_agent, ai_agent1, ai_agent2, ai_agent3]
        self.trainer = Trainer(self.env, self.agents)
        
        print("\n游戏开始！你是玩家0。")
        print("输入对应的数字来选择动作。")
        print("祝你好运！\n")
        
        winner, scores = self.trainer.play_game(render=True)
        
        if winner == 0:
            print("\n🎉 恭喜你获胜！")
        elif winner >= 0:
            print(f"\n😢 玩家 {winner} 获胜了。")
        else:
            print("\n🤝 流局！")
        
        print(f"最终分数: {scores}")
    
    def play_ai_vs_ai(self):
        print("\n" + "="*60)
        print("AI对战模式")
        print("="*60)
        
        num_games = int(input("请输入对战局数 (默认10): ") or "10")
        
        self.env = MahjongEnv(seed=None)
        
        ai1 = RandomAgent("AI_1", seed=42)
        ai2 = RandomAgent("AI_2", seed=43)
        ai3 = RandomAgent("AI_3", seed=44)
        ai4 = RandomAgent("AI_4", seed=45)
        
        self.agents = [ai1, ai2, ai3, ai4]
        self.trainer = Trainer(self.env, self.agents)
        
        print(f"\n开始 {num_games} 局AI对战...")
        results = self.trainer.tournament(num_games)
        
        print("\n" + "="*60)
        print("比赛结果")
        print("="*60)
        print(f"总局数: {results['total_games']}")
        print(f"流局数: {results['draws']}")
        print("\n各玩家胜率:")
        for i, rate in enumerate(results['win_rates']):
            print(f"  玩家{i} ({self.agents[i].name}): {rate*100:.1f}% ({results['wins'][i]}胜)")
        print("\n平均分数:")
        for i, score in enumerate(results['avg_scores']):
            print(f"  玩家{i}: {score:.0f}")
    
    def train_ai(self):
        print("\n" + "="*60)
        print("训练AI模型")
        print("="*60)
        
        num_episodes = int(input("请输入训练回合数 (默认1000): ") or "1000")
        save_freq = int(input("请输入保存频率 (默认100): ") or "100")
        
        print("\n初始化训练环境...")
        self.env = MahjongEnv(seed=42)
        
        dqn_agent = DQNAgent(
            name="DQN_Agent",
            input_shape=(6, 34, 4),
            num_actions=38,
            hidden_size=256,
            learning_rate=0.001,
            seed=42
        )
        
        opponent = RandomAgent("Opponent", seed=43)
        
        self.agents = [dqn_agent, opponent, opponent, opponent]
        self.trainer = Trainer(self.env, self.agents)
        
        print(f"\n开始训练 {num_episodes} 回合...")
        history = self.trainer.train(
            num_episodes=num_episodes,
            save_freq=save_freq,
            eval_freq=save_freq
        )
        
        print("\n训练完成！")
        print(f"训练历史已保存到 logs/ 目录")
    
    def evaluate_ai(self):
        print("\n" + "="*60)
        print("评估AI性能")
        print("="*60)
        
        model_path = input("请输入模型路径 (留空使用随机AI): ").strip()
        
        self.env = MahjongEnv(seed=42)
        
        if model_path:
            agent = DQNAgent(name="Trained_Agent")
            agent.load(model_path)
            agent.set_training(False)
        else:
            agent = RandomAgent("Random_Agent", seed=42)
        
        opponent = RandomAgent("Opponent", seed=43)
        
        self.agents = [agent, opponent, opponent, opponent]
        self.trainer = Trainer(self.env, self.agents)
        
        num_eval = int(input("请输入评估局数 (默认100): ") or "100")
        
        print(f"\n开始评估 {num_eval} 局...")
        results = self.trainer.tournament(num_eval)
        
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        print(f"AI胜率: {results['win_rates'][0]*100:.1f}%")
        print(f"AI平均分数: {results['avg_scores'][0]:.0f}")
        print(f"对手平均胜率: {sum(results['win_rates'][1:])/3*100:.1f}%")
    
    def show_rules(self):
        print("\n" + "="*60)
        print("麻将规则说明")
        print("="*60)
        
        rules_text = """
日本麻将（立直麻将）基本规则：

1. 牌型：
   - 万子 (🀇-🀏): 1-9万
   - 条子 (🀐-🀘): 1-9条
   - 筒子 (🀙-🀡): 1-9筒
   - 风牌 (🀀-🀃): 东南西北
   - 箭牌 (🀄-🀆): 中发白

2. 和牌条件：
   - 4组面子（顺子或刻子）+ 1对雀头
   - 七对子：7个对子
   - 国士无双：13种幺九牌+其中1种成对

3. 基本操作：
   - 摸牌：从牌山摸一张牌
   - 舍牌：打出一张手牌
   - 立直：门前清时宣告听牌
   - 吃：用上家打出的牌组成顺子
   - 碰：用其他玩家打出的牌组成刻子
   - 杠：用4张相同的牌开杠
   - 荣和：和其他玩家打出的牌
   - 自摸：和自己摸到的牌

4. 役种（部分）：
   - 断幺九：不含幺九牌
   - 平和：4组顺子+1对雀头
   - 一杯口：两组相同的顺子
   - 混一色：只有一种数牌+字牌
   - 清一色：只有一种数牌
   - 对对和：4组刻子
   - 三暗刻：3组暗刻

5. 计分：
   - 基本点 = 符 × 2^(番+2)
   - 满贯：5番或基本点2000以上
   - 跳满：6-7番
   - 倍满：8-10番
   - 三倍满：11-12番
   - 役满：13番以上或特殊役
        """
        
        print(rules_text)
        input("\n按回车键返回主菜单...")

def main():
    ui = CLIUI()
    ui.run()

if __name__ == "__main__":
    main()
