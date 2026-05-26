import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.rules import MahjongRules
from src.utils.tile_utils import TileUtils

def debug_yaku_detailed():
    print("="*60)
    print("详细调试役种检测")
    print("="*60)
    
    print("\n--- 测试1: 断幺九平和 (门清, 无立直) ---")
    hand1 = [
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 2),
        TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4), TileUtils.create_tile('wan', 5),
        TileUtils.create_tile('tiao', 3), TileUtils.create_tile('tiao', 4), TileUtils.create_tile('tiao', 5),
        TileUtils.create_tile('tong', 6), TileUtils.create_tile('tong', 7), TileUtils.create_tile('tong', 8),
        TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7), TileUtils.create_tile('tiao', 8)
    ]
    
    symbols = [TileUtils.get_tile_symbol(t) for t in hand1]
    print(f"手牌: {' '.join(symbols)}")
    
    is_win, win_info = MahjongRules.check_win(hand1, [], is_menzen=True, is_riichi=False)
    print(f"和牌检测 (门清, 无立直): {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("❌ 未能和牌")
    
    print("\n--- 测试2: 断幺九平和 (门清, 有立直) ---")
    is_win, win_info = MahjongRules.check_win(hand1, [], is_menzen=True, is_riichi=True)
    print(f"和牌检测 (门清, 有立直): {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("❌ 未能和牌")
    
    print("\n--- 测试3: 检查断幺九检测 ---")
    all_tiles = hand1.copy()
    has_tanyao = all(not TileUtils.is_terminal_tile(t) and not TileUtils.is_honor_tile(t) 
                    for t in all_tiles)
    print(f"是否断幺九: {has_tanyao}")
    
    print("\n--- 测试4: 检查牌型 ---")
    tile_counts = TileUtils.count_tiles(hand1)
    print(f"牌计数: {tile_counts}")
    
    for tile_id, count in tile_counts.items():
        if count >= 2:
            pair_tile = TileUtils.create_tile(*TileUtils.id_to_tile(tile_id))
            print(f"\n尝试将牌 {TileUtils.get_tile_symbol(pair_tile)} 作为对子:")
            
            remaining = []
            pair_removed = 0
            for t in hand1:
                if t['id'] == tile_id and pair_removed < 2:
                    pair_removed += 1
                else:
                    remaining.append(t)
            
            if len(remaining) == 12:
                is_standard, sets = MahjongRules.check_standard_form(remaining)
                print(f"  标准牌型检测: {is_standard}")
                if is_standard:
                    print(f"  面子数: {len(sets)}")
                    for i, s in enumerate(sets):
                        s_symbols = [TileUtils.get_tile_symbol(t) for t in s]
                        print(f"    面子{i}: {' '.join(s_symbols)}")
                    
                    yaku, han, fu = MahjongRules.calculate_yaku_and_han(
                        hand1, [], pair_tile, sets, is_menzen=True, is_riichi=False
                    )
                    print(f"  役种检测 (is_riichi=False): {yaku}")
                    print(f"  番数: {han}")
                    
                    yaku2, han2, fu2 = MahjongRules.calculate_yaku_and_han(
                        hand1, [], pair_tile, sets, is_menzen=True, is_riichi=True
                    )
                    print(f"  役种检测 (is_riichi=True): {yaku2}")
                    print(f"  番数: {han2}")

if __name__ == "__main__":
    debug_yaku_detailed()
