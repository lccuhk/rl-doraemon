import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tile_utils import TileUtils
from src.environment.rules import MahjongRules

def main():
    print("="*70)
    print("手动测试和牌检测逻辑")
    print("="*70)
    
    print("\n【测试1: 简单的平和牌型】")
    print("牌型: 111万 234万 567万 888万 99万")
    
    hand = []
    for _ in range(3):
        hand.append(TileUtils.create_tile('wan', 1))
    for v in [2, 3, 4]:
        hand.append(TileUtils.create_tile('wan', v))
    for v in [5, 6, 7]:
        hand.append(TileUtils.create_tile('wan', v))
    for _ in range(3):
        hand.append(TileUtils.create_tile('wan', 8))
    for _ in range(2):
        hand.append(TileUtils.create_tile('wan', 9))
    
    print(f"手牌数量: {len(hand)}")
    print(f"手牌: {[TileUtils.get_tile_symbol(t) for t in hand]}")
    
    is_win, win_info = MahjongRules.check_win(hand, [], is_menzen=True)
    print(f"是否和牌: {is_win}")
    if is_win:
        print(f"和牌信息: {win_info}")
    else:
        print("【问题】应该和牌但检测失败！")
    
    print("\n" + "="*70)
    print("【测试2: 七对子】")
    print("牌型: 11万 22万 33万 44万 55万 66万 77万")
    
    hand2 = []
    for v in range(1, 8):
        for _ in range(2):
            hand2.append(TileUtils.create_tile('wan', v))
    
    print(f"手牌数量: {len(hand2)}")
    print(f"手牌: {[TileUtils.get_tile_symbol(t) for t in hand2]}")
    
    is_win2, win_info2 = MahjongRules.check_win(hand2, [], is_menzen=True)
    print(f"是否和牌: {is_win2}")
    if is_win2:
        print(f"和牌信息: {win_info2}")
    else:
        print("【问题】应该和牌但检测失败！")
    
    print("\n" + "="*70)
    print("【测试3: 断幺九】")
    print("牌型: 222万 345万 678万 222筒 33筒")
    
    hand3 = []
    for _ in range(3):
        hand3.append(TileUtils.create_tile('wan', 2))
    for v in [3, 4, 5]:
        hand3.append(TileUtils.create_tile('wan', v))
    for v in [6, 7, 8]:
        hand3.append(TileUtils.create_tile('wan', v))
    for _ in range(3):
        hand3.append(TileUtils.create_tile('tong', 2))
    for _ in range(2):
        hand3.append(TileUtils.create_tile('tong', 3))
    
    print(f"手牌数量: {len(hand3)}")
    print(f"手牌: {[TileUtils.get_tile_symbol(t) for t in hand3]}")
    
    is_win3, win_info3 = MahjongRules.check_win(hand3, [], is_menzen=True)
    print(f"是否和牌: {is_win3}")
    if is_win3:
        print(f"和牌信息: {win_info3}")
    else:
        print("【问题】应该和牌但检测失败！")
    
    print("\n" + "="*70)
    print("【测试4: 国士无双】")
    print("牌型: 19万 19筒 19条 东南西北 中发白 + 1万(对子)")
    
    hand4 = [
        TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('wan', 9),
        TileUtils.create_tile('tong', 1),
        TileUtils.create_tile('tong', 9),
        TileUtils.create_tile('tiao', 1),
        TileUtils.create_tile('tiao', 9),
        TileUtils.create_tile('feng', 1),
        TileUtils.create_tile('feng', 2),
        TileUtils.create_tile('feng', 3),
        TileUtils.create_tile('feng', 4),
        TileUtils.create_tile('jian', 1),
        TileUtils.create_tile('jian', 2),
        TileUtils.create_tile('jian', 3),
        TileUtils.create_tile('wan', 1),
    ]
    
    print(f"手牌数量: {len(hand4)}")
    print(f"手牌: {[TileUtils.get_tile_symbol(t) for t in hand4]}")
    
    is_win4, win_info4 = MahjongRules.check_win(hand4, [], is_menzen=True)
    print(f"是否和牌: {is_win4}")
    if is_win4:
        print(f"和牌信息: {win_info4}")
    else:
        print("【问题】应该和牌但检测失败！")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
