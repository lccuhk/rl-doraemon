from typing import List, Dict, Tuple, Optional, Set
from collections import Counter
from ..utils.tile_utils import TileUtils

class MahjongRules:
    YAKU_INFO = {
        'menshin_tsumo': {'han': 1, 'name': '门前清自摸', 'requires_menzen': True},
        'riichi': {'han': 1, 'name': '立直', 'requires_menzen': True},
        'ippatsu': {'han': 1, 'name': '一发', 'requires_menzen': True},
        'tanyao': {'han': 1, 'name': '断幺九', 'requires_menzen': False},
        'pinfu': {'han': 1, 'name': '平和', 'requires_menzen': True},
        'ipeikou': {'han': 1, 'name': '一杯口', 'requires_menzen': True},
        'yakuhai': {'han': 1, 'name': '役牌', 'requires_menzen': False},
        'ton_nan_pei_sha': {'han': 1, 'name': '场风/自风', 'requires_menzen': False},
        'chankan': {'han': 1, 'name': '抢杠', 'requires_menzen': False},
        'rinshan_kaihou': {'han': 1, 'name': '岭上开花', 'requires_menzen': False},
        'haitei_raoyue': {'han': 1, 'name': '海底摸月', 'requires_menzen': False},
        'houtei_raoyui': {'han': 1, 'name': '河底捞鱼', 'requires_menzen': False},
        'sanshoku_doujun': {'han': 2, 'name': '三色同顺', 'requires_menzen': False, 'open_penalty': True},
        'ittsu': {'han': 2, 'name': '一气通贯', 'requires_menzen': False, 'open_penalty': True},
        'toitoi': {'han': 2, 'name': '对对和', 'requires_menzen': False},
        'sanankou': {'han': 2, 'name': '三暗刻', 'requires_menzen': False},
        'sanshoku_doukou': {'han': 2, 'name': '三色同刻', 'requires_menzen': False},
        'sankantsu': {'han': 2, 'name': '三杠子', 'requires_menzen': False},
        'chiitoitsu': {'han': 2, 'name': '七对子', 'requires_menzen': True},
        'honroutou': {'han': 2, 'name': '混老头', 'requires_menzen': False},
        'shousangen': {'han': 2, 'name': '小三元', 'requires_menzen': False},
        'honitsu': {'han': 3, 'name': '混一色', 'requires_menzen': False, 'open_penalty': True},
        'junchan': {'han': 3, 'name': '纯全带幺九', 'requires_menzen': False, 'open_penalty': True},
        'ryanpeikou': {'han': 3, 'name': '二杯口', 'requires_menzen': True},
        'chinitsu': {'han': 6, 'name': '清一色', 'requires_menzen': False, 'open_penalty': True},
    }
    
    YAKUMAN_INFO = {
        'kokushi_musou': {'name': '国士无双', 'is_double': False},
        'suuankou': {'name': '四暗刻', 'is_double': False},
        'daisangen': {'name': '大三元', 'is_double': False},
        'tsuuiisou': {'name': '字一色', 'is_double': False},
        'ryuuiisou': {'name': '绿一色', 'is_double': False},
        'chinroutou': {'name': '清老头', 'is_double': False},
        'shousuushii': {'name': '小四喜', 'is_double': False},
        'daisuushii': {'name': '大四喜', 'is_double': True},
        'suukantsu': {'name': '四杠子', 'is_double': False},
        'tenhou': {'name': '天和', 'is_double': False},
        'chiihou': {'name': '地和', 'is_double': False},
        'kokushi_musou_13': {'name': '国士无双十三面', 'is_double': True},
        'suuankou_tanki': {'name': '四暗刻单骑', 'is_double': True},
    }
    
    @staticmethod
    def check_chi(hand: List[Dict], tile: Dict) -> List[List[Dict]]:
        if TileUtils.is_honor_tile(tile):
            return []
        
        possible_chis = []
        tile_type = tile['type']
        value = tile['value']
        
        hand_counts = TileUtils.count_tiles(hand)
        
        for start in range(max(1, value - 2), min(value + 1, 8)):
            needed = [start, start + 1, start + 2]
            if value in needed:
                needed_without_tile = [v for v in needed if v != value]
                can_make = True
                for v in needed_without_tile:
                    needed_id = TileUtils.tile_to_id(tile_type, v)
                    if hand_counts.get(needed_id, 0) < 1:
                        can_make = False
                        break
                if can_make:
                    chi_tiles = [TileUtils.create_tile(tile_type, v) for v in needed]
                    possible_chis.append(chi_tiles)
        
        return possible_chis
    
    @staticmethod
    def check_pon(hand: List[Dict], tile: Dict) -> bool:
        hand_counts = TileUtils.count_tiles(hand)
        return hand_counts.get(tile['id'], 0) >= 2
    
    @staticmethod
    def check_kan(hand: List[Dict], tile: Dict = None) -> List[Dict]:
        hand_counts = TileUtils.count_tiles(hand)
        kans = []
        
        if tile is not None:
            if hand_counts.get(tile['id'], 0) >= 3:
                kans.append(tile)
        else:
            for tile_id, count in hand_counts.items():
                if count >= 4:
                    tile_type, value = TileUtils.id_to_tile(tile_id)
                    kans.append(TileUtils.create_tile(tile_type, value))
        
        return kans
    
    @staticmethod
    def check_ron(hand: List[Dict], tile: Dict, melds: List[List[Dict]] = None, 
                  is_menzen: bool = True, is_riichi: bool = False) -> Tuple[bool, Dict]:
        if melds is None:
            melds = []
        
        test_hand = hand + [tile]
        
        is_win, win_info = MahjongRules.check_win(test_hand, melds, is_menzen, is_riichi)
        
        if is_win:
            win_info['win_type'] = 'ron'
            win_info['win_tile'] = tile
        
        return is_win, win_info
    
    @staticmethod
    def check_tsumo(hand: List[Dict], melds: List[List[Dict]] = None,
                    is_menzen: bool = True, is_riichi: bool = False) -> Tuple[bool, Dict]:
        if melds is None:
            melds = []
        
        is_win, win_info = MahjongRules.check_win(hand, melds, is_menzen, is_riichi)
        
        if is_win:
            win_info['win_type'] = 'tsumo'
        
        return is_win, win_info
    
    @staticmethod
    def check_win(hand: List[Dict], melds: List[List[Dict]], is_menzen: bool, is_riichi: bool = False) -> Tuple[bool, Dict]:
        if len(hand) != 14:
            return False, {}
        
        all_tiles = hand.copy()
        for meld in melds:
            all_tiles.extend(meld)
        
        is_kokushi, kokushi_info = MahjongRules.check_kokushi_musou(hand)
        if is_kokushi:
            return True, {'yaku': ['kokushi_musou'], 'han': 13, 'fu': 0, 'is_yakuman': True}
        
        is_chiitoitsu, chiitoitsu_info = MahjongRules.check_chiitoitsu(hand)
        if is_chiitoitsu:
            yaku = ['chiitoitsu']
            han = 2
            fu = 25
            return True, {'yaku': yaku, 'han': han, 'fu': fu, 'is_yakuman': False}
        
        tile_counts = TileUtils.count_tiles(hand)
        
        for tile_id, count in tile_counts.items():
            if count >= 2:
                pair_tile = TileUtils.create_tile(*TileUtils.id_to_tile(tile_id))
                
                remaining = []
                pair_removed = 0
                for t in hand:
                    if t['id'] == tile_id and pair_removed < 2:
                        pair_removed += 1
                    else:
                        remaining.append(t)
                
                if len(remaining) == 12:
                    is_standard, sets = MahjongRules.check_standard_form(remaining)
                    if is_standard:
                        yaku, han, fu = MahjongRules.calculate_yaku_and_han(
                            hand, melds, pair_tile, sets, is_menzen, is_riichi
                        )
                        if yaku:
                            return True, {
                                'yaku': yaku, 
                                'han': han, 
                                'fu': fu, 
                                'is_yakuman': False,
                                'pair': pair_tile,
                                'sets': sets
                            }
        
        return False, {}
    
    @staticmethod
    def check_kokushi_musou(hand: List[Dict]) -> Tuple[bool, Dict]:
        if len(hand) != 14:
            return False, {}
        
        terminal_ids = set()
        for tile_type in ['wan', 'tiao', 'tong']:
            terminal_ids.add(TileUtils.tile_to_id(tile_type, 1))
            terminal_ids.add(TileUtils.tile_to_id(tile_type, 9))
        
        for tile_type in ['feng']:
            for v in range(1, 5):
                terminal_ids.add(TileUtils.tile_to_id(tile_type, v))
        
        for tile_type in ['jian']:
            for v in range(1, 4):
                terminal_ids.add(TileUtils.tile_to_id(tile_type, v))
        
        hand_counts = TileUtils.count_tiles(hand)
        
        has_all_terminals = all(tid in hand_counts for tid in terminal_ids)
        if not has_all_terminals:
            return False, {}
        
        pair_count = sum(1 for tid, cnt in hand_counts.items() if cnt >= 2)
        if pair_count == 1:
            return True, {'type': 'normal'}
        elif pair_count == 0 and len(hand_counts) == 13:
            return True, {'type': '13_wait'}
        
        return False, {}
    
    @staticmethod
    def check_chiitoitsu(hand: List[Dict]) -> Tuple[bool, Dict]:
        if len(hand) != 14:
            return False, {}
        
        hand_counts = TileUtils.count_tiles(hand)
        
        pairs = [tid for tid, cnt in hand_counts.items() if cnt == 2]
        
        if len(pairs) == 7 and len(hand_counts) == 7:
            return True, {'pairs': pairs}
        
        return False, {}
    
    @staticmethod
    def check_standard_form(tiles: List[Dict]) -> Tuple[bool, List[List[Dict]]]:
        if len(tiles) == 0:
            return True, []
        
        tile_counts = TileUtils.count_tiles(tiles)
        tile_ids = sorted(tile_counts.keys())
        
        if not tile_ids:
            return True, []
        
        first_tid = tile_ids[0]
        tile_type, value = TileUtils.id_to_tile(first_tid)
        
        if tile_counts[first_tid] >= 3:
            new_counts = tile_counts.copy()
            new_counts[first_tid] -= 3
            if new_counts[first_tid] == 0:
                del new_counts[first_tid]
            
            remaining_tiles = []
            for tid, cnt in new_counts.items():
                tt, v = TileUtils.id_to_tile(tid)
                remaining_tiles.extend([TileUtils.create_tile(tt, v)] * cnt)
            
            is_valid, sets = MahjongRules.check_standard_form(remaining_tiles)
            if is_valid:
                triplet = [TileUtils.create_tile(tile_type, value)] * 3
                return True, [triplet] + sets
        
        if tile_type in ['wan', 'tiao', 'tong']:
            if value <= 7:
                seq_tids = [
                    TileUtils.tile_to_id(tile_type, value),
                    TileUtils.tile_to_id(tile_type, value + 1),
                    TileUtils.tile_to_id(tile_type, value + 2)
                ]
                if all(tid in tile_counts for tid in seq_tids):
                    new_counts = tile_counts.copy()
                    for tid in seq_tids:
                        new_counts[tid] -= 1
                        if new_counts[tid] == 0:
                            del new_counts[tid]
                    
                    remaining_tiles = []
                    for tid, cnt in new_counts.items():
                        tt, v = TileUtils.id_to_tile(tid)
                        remaining_tiles.extend([TileUtils.create_tile(tt, v)] * cnt)
                    
                    is_valid, sets = MahjongRules.check_standard_form(remaining_tiles)
                    if is_valid:
                        sequence = [TileUtils.create_tile(tile_type, value + i) for i in range(3)]
                        return True, [sequence] + sets
        
        return False, []
    
    @staticmethod
    def calculate_yaku_and_han(hand: List[Dict], melds: List[List[Dict]], 
                                pair: Dict, sets: List[List[Dict]], 
                                is_menzen: bool, is_riichi: bool = False) -> Tuple[List[str], int, int]:
        yaku = []
        total_han = 0
        fu = 20
        
        if is_riichi and is_menzen:
            yaku.append('riichi')
            total_han += 1
        
        all_tiles = hand.copy()
        for meld in melds:
            all_tiles.extend(meld)
        
        tile_counts = TileUtils.count_tiles(all_tiles)
        
        has_tanyao = all(not TileUtils.is_terminal_tile(t) and not TileUtils.is_honor_tile(t) 
                        for t in all_tiles)
        if has_tanyao:
            yaku.append('tanyao')
            total_han += 1
        
        is_honitsu = False
        is_chinitsu = False
        
        tile_types = set(t['type'] for t in all_tiles)
        has_honor = any(TileUtils.is_honor_tile(t) for t in all_tiles)
        
        if len(tile_types) == 1:
            is_chinitsu = True
        elif len(tile_types) == 2 and has_honor:
            is_honitsu = True
        
        if is_chinitsu:
            yaku.append('chinitsu')
            total_han += 5 if not is_menzen else 6
        elif is_honitsu:
            yaku.append('honitsu')
            total_han += 2 if not is_menzen else 3
        
        all_sets = sets + melds
        triplet_count = 0
        sequence_count = 0
        
        for s in all_sets:
            if len(s) == 3:
                if s[0]['id'] == s[1]['id'] == s[2]['id']:
                    triplet_count += 1
                else:
                    sequence_count += 1
        
        if triplet_count == 4:
            yaku.append('toitoi')
            total_han += 2
        
        if is_menzen and sequence_count >= 2:
            yaku.append('ipeikou')
            total_han += 1
        
        if not yaku:
            return [], 0, 0
        
        return yaku, total_han, fu
    
    @staticmethod
    def calculate_score(han: int, fu: int, is_tsumo: bool, is_dealer: bool,
                        is_yakuman: bool = False, yakuman_count: int = 1) -> Dict:
        if is_yakuman:
            base_score = 8000 * yakuman_count
        else:
            if han >= 13:
                base_score = 8000
            elif han >= 11:
                base_score = 6000
            elif han >= 8:
                base_score = 4000
            elif han >= 6:
                base_score = 3000
            else:
                base_score = fu * (2 ** (han + 2))
                if base_score > 2000:
                    base_score = 2000
        
        if is_dealer:
            if is_tsumo:
                each_payment = (base_score * 2 + 99) // 100 * 100
                total_win = each_payment * 3
            else:
                total_win = (base_score * 6 + 99) // 100 * 100
                each_payment = total_win
        else:
            if is_tsumo:
                dealer_payment = (base_score * 2 + 99) // 100 * 100
                other_payment = (base_score + 99) // 100 * 100
                total_win = dealer_payment + other_payment * 2
                each_payment = {'dealer': dealer_payment, 'others': other_payment}
            else:
                total_win = (base_score * 4 + 99) // 100 * 100
                each_payment = total_win
        
        return {
            'base_score': base_score,
            'total_win': total_win,
            'each_payment': each_payment,
            'is_dealer': is_dealer,
            'is_tsumo': is_tsumo
        }
    
    @staticmethod
    def get_legal_discards(hand: List[Dict]) -> List[int]:
        return list(range(len(hand)))
    
    @staticmethod
    def get_shanten(hand: List[Dict], melds: List[List[Dict]] = None) -> int:
        if melds is None:
            melds = []
        
        base_shanten = 8
        
        tile_counts = TileUtils.count_tiles(hand)
        
        pairs = sum(1 for cnt in tile_counts.values() if cnt >= 2)
        triplets = sum(1 for cnt in tile_counts.values() if cnt >= 3)
        quads = sum(1 for cnt in tile_counts.values() if cnt >= 4)
        
        potential_sets = triplets + quads
        shanten = base_shanten - pairs - potential_sets * 2
        
        return max(0, shanten)
