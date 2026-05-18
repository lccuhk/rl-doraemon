from typing import Dict, List
from .base_agent import BaseAgent
from ..utils.tile_utils import TileUtils
from ..environment.mahjong_env import ActionType, GamePhase

class HumanAgent(BaseAgent):
    def __init__(self, name: str = "HumanAgent"):
        super().__init__(name)
    
    def select_action(self, observation: Dict, legal_actions: List[int]) -> int:
        print(f"\n{'='*60}")
        print(f"轮到你行动了！")
        print(f"{'='*60}")
        
        hand = observation.get('hand', [])
        phase = observation.get('phase', 0)
        
        print(f"\n当前阶段: {GamePhase(phase).name}")
        print(f"\n你的手牌:")
        for i, tile in enumerate(hand):
            symbol = TileUtils.get_tile_symbol(tile)
            name = TileUtils.get_tile_name(tile)
            print(f"  [{i}] {symbol} {name}")
        
        print(f"\n可选动作:")
        action_descriptions = self._get_action_descriptions(legal_actions, observation, phase)
        
        for i, (action, desc) in enumerate(action_descriptions):
            print(f"  [{i}] {desc}")
        
        while True:
            try:
                choice = input(f"\n请选择动作 (0-{len(action_descriptions)-1}): ")
                choice_idx = int(choice)
                if 0 <= choice_idx < len(action_descriptions):
                    return action_descriptions[choice_idx][0]
                else:
                    print(f"请输入 0 到 {len(action_descriptions)-1} 之间的数字")
            except ValueError:
                print("请输入有效的数字")
    
    def _get_action_descriptions(self, legal_actions: List[int], 
                                  observation: Dict, phase: int) -> List[tuple]:
        descriptions = []
        hand = observation.get('hand', [])
        
        for action in legal_actions:
            if phase == GamePhase.DRAWING.value:
                if action == ActionType.TSUMO.value:
                    descriptions.append((action, "自摸胡牌"))
                elif action == ActionType.RIICHI.value:
                    descriptions.append((action, "立直"))
                elif action == ActionType.KAN.value:
                    descriptions.append((action, "暗杠"))
                else:
                    descriptions.append((action, "进入舍牌阶段"))
            elif phase == GamePhase.DISCARDING.value:
                if 0 <= action < len(hand):
                    tile = hand[action]
                    symbol = TileUtils.get_tile_symbol(tile)
                    name = TileUtils.get_tile_name(tile)
                    descriptions.append((action, f"舍牌: {symbol} {name}"))
            elif phase == GamePhase.RESPONDING.value:
                if action == ActionType.RON.value:
                    descriptions.append((action, "荣和"))
                elif action == ActionType.PON.value:
                    descriptions.append((action, "碰"))
                elif action == ActionType.CHI.value:
                    descriptions.append((action, "吃"))
                elif action == ActionType.KAN.value:
                    descriptions.append((action, "明杠"))
                elif action == ActionType.PASS.value:
                    descriptions.append((action, "跳过"))
        
        return descriptions
