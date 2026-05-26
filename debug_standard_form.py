import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.rules import MahjongRules
from src.utils.tile_utils import TileUtils

def debug_standard_form():
    print("="*60)
    print("调试标准牌型检测")
    print("="*60)
    
    print("\n--- 测试牌型: 123万 456条 789筒 东南西 11万 ---")
    hand = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7),
        TileUtils.create_tile('tong', 8), TileUtils.create_tile('tong', 9), TileUtils.create_tile('tong', 1),
        TileUtils.create_tile('feng', 1), TileUtils.create_tile('feng', 2), TileUtils.create_tile('feng', 3)
    ]
    
    symbols = [TileUtils.get_tile_symbol(t) for t in hand]
    print(f"完整手牌: {' '.join(symbols)}")
    
    print("\n--- 去掉对子 (11万) 后 ---")
    remaining = hand[2:]
    remaining_symbols = [TileUtils.get_tile_symbol(t) for t in remaining]
    print(f"剩余牌: {' '.join(remaining_symbols)}")
    print(f"剩余牌数: {len(remaining)}")
    
    tile_counts = TileUtils.count_tiles(remaining)
    print(f"牌计数: {tile_counts}")
    
    print("\n--- 手动分析 ---")
    print("应该能组成:")
    print("  1. 234万 (顺子)")
    print("  2. 567条 (顺子)")
    print("  3. 891筒 (顺子) - 注意: 9和1不是连续的!")
    print("  4. 东南西 (字牌顺子) - 注意: 字牌不能组成顺子!")
    
    print("\n--- 问题分析 ---")
    print("1. 筒子: 8, 9, 1 不能组成顺子 (9和1不连续)")
    print("2. 字牌: 东, 南, 西 不能组成顺子 (字牌只能刻子)")
    
    print("\n--- 测试一个能组成顺子的牌型 ---")
    good_hand = [
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7),
        TileUtils.create_tile('tong', 6), TileUtils.create_tile('tong', 7), TileUtils.create_tile('tong', 8),
        TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7), TileUtils.create_tile('tiao', 8)
    ]
    good_symbols = [TileUtils.get_tile_symbol(t) for t in good_hand]
    print(f"测试牌型: {' '.join(good_symbols)}")
    
    is_standard, sets = MahjongRules.check_standard_form(good_hand)
    print(f"标准牌型检测: {is_standard}")
    if is_standard:
        print(f"面子数: {len(sets)}")
        for i, s in enumerate(sets):
            s_symbols = [TileUtils.get_tile_symbol(t) for t in s]
            print(f"  面子{i}: {' '.join(s_symbols)}")
    
    print("\n--- 测试带刻子的牌型 ---")
    triplet_hand = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7),
        TileUtils.create_tile('tong', 6), TileUtils.create_tile('tong', 7), TileUtils.create_tile('tong', 8),
        TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7), TileUtils.create_tile('tiao', 8)
    ]
    triplet_symbols = [TileUtils.get_tile_symbol(t) for t in triplet_hand]
    print(f"测试牌型: {' '.join(triplet_symbols)}")
    
    is_standard, sets = MahjongRules.check_standard_form(triplet_hand)
    print(f"标准牌型检测: {is_standard}")
    if is_standard:
        print(f"面子数: {len(sets)}")
        for i, s in enumerate(sets):
            s_symbols = [TileUtils.get_tile_symbol(t) for t in s]
            print(f"  面子{i}: {' '.join(s_symbols)}")

if __name__ == "__main__":
    debug_standard_form()
