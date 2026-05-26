import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.rules import MahjongRules
from src.utils.tile_utils import TileUtils

def debug_yaku():
    print("="*60)
    print("调试役种检测逻辑")
    print("="*60)
    
    print("\n--- 测试: 立直牌型 ---")
    hand = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7),
        TileUtils.create_tile('tong', 8), TileUtils.create_tile('tong', 9), TileUtils.create_tile('tong', 1),
        TileUtils.create_tile('feng', 1), TileUtils.create_tile('feng', 2), TileUtils.create_tile('feng', 3)
    ]
    
    symbols = [TileUtils.get_tile_symbol(t) for t in hand]
    print(f"手牌: {' '.join(symbols)}")
    print(f"手牌数: {len(hand)}")
    
    print("\n--- 检查标准牌型 ---")
    tile_counts = TileUtils.count_tiles(hand)
    print(f"牌计数: {tile_counts}")
    
    for tile_id, count in tile_counts.items():
        if count >= 2:
            pair_tile = TileUtils.create_tile(*TileUtils.id_to_tile(tile_id))
            print(f"\n尝试将牌 {TileUtils.get_tile_symbol(pair_tile)} 作为对子:")
            
            remaining = []
            pair_removed = 0
            for t in hand:
                if t['id'] == tile_id and pair_removed < 2:
                    pair_removed += 1
                else:
                    remaining.append(t)
            
            print(f"  剩余牌数: {len(remaining)}")
            
            if len(remaining) == 12:
                is_standard, sets = MahjongRules.check_standard_form(remaining)
                print(f"  标准牌型检测: {is_standard}")
                if is_standard:
                    print(f"  面子数: {len(sets)}")
                    for i, s in enumerate(sets):
                        s_symbols = [TileUtils.get_tile_symbol(t) for t in s]
                        print(f"    面子{i}: {' '.join(s_symbols)}")
                    
                    print("\n  --- 计算役种 ---")
                    yaku, han, fu = MahjongRules.calculate_yaku_and_han(
                        hand, [], pair_tile, sets, is_menzen=True, is_riichi=True
                    )
                    print(f"  立直=True, 门清=True")
                    print(f"  检测到的役种: {yaku}")
                    print(f"  番数: {han}")
                    print(f"  符数: {fu}")
                    
                    if not yaku:
                        print("\n  ❌ 问题: 没有检测到任何役种！")
                        print("  分析:")
                        print(f"    - is_riichi=True, is_menzen=True")
                        print(f"    - 立直条件: is_riichi and is_menzen = {True and True}")
    
    print("\n" + "="*60)
    print("直接调用 check_win 测试")
    print("="*60)
    
    is_win, win_info = MahjongRules.check_win(hand, [], is_menzen=True, is_riichi=True)
    print(f"和牌检测结果: {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("❌ 未能和牌")

if __name__ == "__main__":
    debug_yaku()
