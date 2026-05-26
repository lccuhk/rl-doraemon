import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.environment.rules import MahjongRules
from src.utils.tile_utils import TileUtils

def test_win_detection():
    print("="*60)
    print("测试和牌检测逻辑")
    print("="*60)
    
    print("\n--- 测试1: 断幺九平和 (应该能和牌) ---")
    hand1 = [
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 2),
        TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4), TileUtils.create_tile('wan', 5),
        TileUtils.create_tile('tiao', 3), TileUtils.create_tile('tiao', 4), TileUtils.create_tile('tiao', 5),
        TileUtils.create_tile('tong', 6), TileUtils.create_tile('tong', 7), TileUtils.create_tile('tong', 8),
        TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7), TileUtils.create_tile('tiao', 8)
    ]
    print(f"手牌数: {len(hand1)}")
    symbols = [TileUtils.get_tile_symbol(t) for t in hand1]
    print(f"手牌: {' '.join(symbols)}")
    
    is_win, win_info = MahjongRules.check_win(hand1, [], is_menzen=True, is_riichi=False)
    print(f"和牌检测结果: {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("❌ 未能检测到和牌！")
    
    print("\n--- 测试2: 立直 (应该能和牌) ---")
    hand2 = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7),
        TileUtils.create_tile('tong', 8), TileUtils.create_tile('tong', 9), TileUtils.create_tile('tong', 1),
        TileUtils.create_tile('feng', 1), TileUtils.create_tile('feng', 2), TileUtils.create_tile('feng', 3)
    ]
    print(f"手牌数: {len(hand2)}")
    symbols = [TileUtils.get_tile_symbol(t) for t in hand2]
    print(f"手牌: {' '.join(symbols)}")
    
    is_win, win_info = MahjongRules.check_win(hand2, [], is_menzen=True, is_riichi=True)
    print(f"和牌检测结果 (立直): {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("❌ 未能检测到和牌！")
    
    print("\n--- 测试3: 七对子 (应该能和牌) ---")
    hand3 = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 3),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 5),
        TileUtils.create_tile('tong', 7), TileUtils.create_tile('tong', 7),
        TileUtils.create_tile('feng', 1), TileUtils.create_tile('feng', 1),
        TileUtils.create_tile('jian', 1), TileUtils.create_tile('jian', 1),
        TileUtils.create_tile('wan', 9), TileUtils.create_tile('wan', 9)
    ]
    print(f"手牌数: {len(hand3)}")
    symbols = [TileUtils.get_tile_symbol(t) for t in hand3]
    print(f"手牌: {' '.join(symbols)}")
    
    is_win, win_info = MahjongRules.check_win(hand3, [], is_menzen=True, is_riichi=False)
    print(f"和牌检测结果: {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("❌ 未能检测到和牌！")
    
    print("\n--- 测试4: 国士无双 (应该能和牌) ---")
    hand4 = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 9),
        TileUtils.create_tile('tiao', 1), TileUtils.create_tile('tiao', 9),
        TileUtils.create_tile('tong', 1), TileUtils.create_tile('tong', 9),
        TileUtils.create_tile('feng', 1), TileUtils.create_tile('feng', 2),
        TileUtils.create_tile('feng', 3), TileUtils.create_tile('feng', 4),
        TileUtils.create_tile('jian', 1), TileUtils.create_tile('jian', 2),
        TileUtils.create_tile('jian', 3), TileUtils.create_tile('wan', 1)
    ]
    print(f"手牌数: {len(hand4)}")
    symbols = [TileUtils.get_tile_symbol(t) for t in hand4]
    print(f"手牌: {' '.join(symbols)}")
    
    is_win, win_info = MahjongRules.check_win(hand4, [], is_menzen=True, is_riichi=False)
    print(f"和牌检测结果: {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
        print(f"役满: {win_info.get('is_yakuman', False)}")
    else:
        print("❌ 未能检测到和牌！")
    
    print("\n--- 测试5: 无役 (不应该和牌) ---")
    hand5 = [
        TileUtils.create_tile('wan', 1), TileUtils.create_tile('wan', 1),
        TileUtils.create_tile('wan', 2), TileUtils.create_tile('wan', 3), TileUtils.create_tile('wan', 4),
        TileUtils.create_tile('tiao', 5), TileUtils.create_tile('tiao', 6), TileUtils.create_tile('tiao', 7),
        TileUtils.create_tile('tong', 8), TileUtils.create_tile('tong', 9), TileUtils.create_tile('tong', 1),
        TileUtils.create_tile('feng', 1), TileUtils.create_tile('feng', 2), TileUtils.create_tile('feng', 3)
    ]
    print(f"手牌数: {len(hand5)}")
    symbols = [TileUtils.get_tile_symbol(t) for t in hand5]
    print(f"手牌: {' '.join(symbols)}")
    
    is_win, win_info = MahjongRules.check_win(hand5, [], is_menzen=True, is_riichi=False)
    print(f"和牌检测结果 (无役): {is_win}")
    if is_win:
        print(f"役种: {win_info.get('yaku', [])}")
        print(f"番数: {win_info.get('han', 0)}")
    else:
        print("✓ 正确：无役不能和牌")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_win_detection()
