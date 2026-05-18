import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.tile_utils import TileUtils
from src.environment.rules import MahjongRules

def main():
    print("="*70)
    print("详细调试和牌检测逻辑")
    print("="*70)
    
    print("\n【测试: 断幺九】")
    print("牌型: 222万 345万 678万 222筒 33筒")
    
    hand = []
    for _ in range(3):
        hand.append(TileUtils.create_tile('wan', 2))
    for v in [3, 4, 5]:
        hand.append(TileUtils.create_tile('wan', v))
    for v in [6, 7, 8]:
        hand.append(TileUtils.create_tile('wan', v))
    for _ in range(3):
        hand.append(TileUtils.create_tile('tong', 2))
    for _ in range(2):
        hand.append(TileUtils.create_tile('tong', 3))
    
    print(f"手牌数量: {len(hand)}")
    print(f"手牌: {[TileUtils.get_tile_symbol(t) for t in hand]}")
    
    tile_counts = TileUtils.count_tiles(hand)
    print(f"\n牌计数: {tile_counts}")
    
    print("\n--- 检查每个可能的对子 ---")
    for tile_id, count in tile_counts.items():
        if count >= 2:
            tile_type, value = TileUtils.id_to_tile(tile_id)
            pair_tile = TileUtils.create_tile(tile_type, value)
            print(f"\n对子: {TileUtils.get_tile_symbol(pair_tile)} (id={tile_id}, count={count})")
            
            remaining = hand.copy()
            pair_removed = 0
            for i, t in enumerate(remaining):
                if t['id'] == tile_id and pair_removed < 2:
                    remaining.pop(i)
                    pair_removed += 1
                    if pair_removed == 2:
                        break
            
            print(f"剩余牌数量: {len(remaining)}")
            print(f"剩余牌: {[TileUtils.get_tile_symbol(t) for t in remaining]}")
            
            if len(remaining) == 12:
                is_standard, sets = MahjongRules.check_standard_form(remaining)
                print(f"标准形式检测: {is_standard}")
                if is_standard:
                    print(f"面子: {[[TileUtils.get_tile_symbol(t) for t in s] for s in sets]}")
                    
                    yaku, han, fu = MahjongRules.calculate_yaku_and_han(
                        hand, [], pair_tile, sets, is_menzen=True
                    )
                    print(f"役: {yaku}, 番: {han}, 符: {fu}")
                else:
                    print("【问题】标准形式检测失败！")
                    
                    print("\n--- 手动检查剩余牌 ---")
                    rem_counts = TileUtils.count_tiles(remaining)
                    print(f"剩余牌计数: {rem_counts}")
                    
                    print("\n尝试分解:")
                    rem_tile_ids = sorted(rem_counts.keys())
                    for tid in rem_tile_ids:
                        tt, v = TileUtils.id_to_tile(tid)
                        cnt = rem_counts[tid]
                        print(f"  {TileUtils.get_tile_symbol(TileUtils.create_tile(tt, v))}: {cnt}张")
                        
                        if cnt >= 3:
                            print(f"    -> 可以组成刻子")
                        
                        if tt in ['wan', 'tiao', 'tong']:
                            seq_ids = [
                                TileUtils.tile_to_id(tt, v),
                                TileUtils.tile_to_id(tt, v+1) if v+1 <= 9 else -1,
                                TileUtils.tile_to_id(tt, v+2) if v+2 <= 9 else -1
                            ]
                            if all(sid in rem_counts for sid in seq_ids if sid >= 0):
                                print(f"    -> 可以组成顺子 {v}-{v+1}-{v+2}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
