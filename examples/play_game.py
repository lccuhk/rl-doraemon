import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.mahjong_env import MahjongEnv
from src.agents.random_agent import RandomAgent
from src.agents.human_agent import HumanAgent
from src.training.trainer import Trainer

def main():
    print("="*60)
    print("麻将AI - 游戏示例")
    print("="*60)
    
    print("\n选择游戏模式:")
    print("1. 人机对战")
    print("2. AI对战演示")
    
    choice = input("\n请选择 (1-2): ").strip()
    
    env = MahjongEnv(seed=None)
    
    if choice == '1':
        human_agent = HumanAgent("Human")
        ai_agent1 = RandomAgent("AI_1", seed=42)
        ai_agent2 = RandomAgent("AI_2", seed=43)
        ai_agent3 = RandomAgent("AI_3", seed=44)
        
        agents = [human_agent, ai_agent1, ai_agent2, ai_agent3]
        trainer = Trainer(env, agents)
        
        print("\n游戏开始！你是玩家0。")
        print("输入对应的数字来选择动作。")
        print("祝你好运！\n")
        
        winner, scores = trainer.play_game(render=True)
        
        if winner == 0:
            print("\n🎉 恭喜你获胜！")
        elif winner >= 0:
            print(f"\n😢 玩家 {winner} 获胜了。")
        else:
            print("\n🤝 流局！")
        
        print(f"最终分数: {scores}")
    
    else:
        ai1 = RandomAgent("AI_1", seed=42)
        ai2 = RandomAgent("AI_2", seed=43)
        ai3 = RandomAgent("AI_3", seed=44)
        ai4 = RandomAgent("AI_4", seed=45)
        
        agents = [ai1, ai2, ai3, ai4]
        trainer = Trainer(env, agents)
        
        print("\nAI对战演示开始...\n")
        winner, scores = trainer.play_game(render=True)
        
        if winner >= 0:
            print(f"\n🏆 玩家 {winner} ({agents[winner].name}) 获胜！")
        else:
            print("\n🤝 流局！")
        
        print(f"最终分数: {scores}")

if __name__ == "__main__":
    main()
