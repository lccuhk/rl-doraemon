import numpy as np
from typing import List, Dict, Tuple, Optional

class TileUtils:
    TILE_TYPES = ['wan', 'tiao', 'tong', 'feng', 'jian']
    
    TILE_NAMES = {
        'wan': ['一万', '二万', '三万', '四万', '五万', '六万', '七万', '八万', '九万'],
        'tiao': ['一条', '二条', '三条', '四条', '五条', '六条', '七条', '八条', '九条'],
        'tong': ['一筒', '二筒', '三筒', '四筒', '五筒', '六筒', '七筒', '八筒', '九筒'],
        'feng': ['东', '南', '西', '北'],
        'jian': ['中', '发', '白']
    }
    
    TILE_SYMBOLS = {
        'wan': ['🀇', '🀈', '🀉', '🀊', '🀋', '🀌', '🀍', '🀎', '🀏'],
        'tiao': ['🀐', '🀑', '🀒', '🀓', '🀔', '🀕', '🀖', '🀗', '🀘'],
        'tong': ['🀙', '🀚', '🀛', '🀜', '🀝', '🀞', '🀟', '🀠', '🀡'],
        'feng': ['🀀', '🀁', '🀂', '🀃'],
        'jian': ['🀄', '🀅', '🀆']
    }
    
    @staticmethod
    def create_tile(tile_type: str, value: int) -> Dict:
        return {
            'type': tile_type,
            'value': value,
            'id': TileUtils.tile_to_id(tile_type, value)
        }
    
    @staticmethod
    def tile_to_id(tile_type: str, value: int) -> int:
        if tile_type == 'wan':
            return value - 1
        elif tile_type == 'tiao':
            return 9 + value - 1
        elif tile_type == 'tong':
            return 18 + value - 1
        elif tile_type == 'feng':
            return 27 + value - 1
        elif tile_type == 'jian':
            return 31 + value - 1
        return -1
    
    @staticmethod
    def id_to_tile(tile_id: int) -> Tuple[str, int]:
        if tile_id < 9:
            return ('wan', tile_id + 1)
        elif tile_id < 18:
            return ('tiao', tile_id - 9 + 1)
        elif tile_id < 27:
            return ('tong', tile_id - 18 + 1)
        elif tile_id < 31:
            return ('feng', tile_id - 27 + 1)
        elif tile_id < 34:
            return ('jian', tile_id - 31 + 1)
        return ('', 0)
    
    @staticmethod
    def get_tile_name(tile: Dict) -> str:
        tile_type = tile['type']
        value = tile['value']
        return TileUtils.TILE_NAMES[tile_type][value - 1]
    
    @staticmethod
    def get_tile_symbol(tile: Dict) -> str:
        tile_type = tile['type']
        value = tile['value']
        return TileUtils.TILE_SYMBOLS[tile_type][value - 1]
    
    @staticmethod
    def create_full_wall() -> List[Dict]:
        wall = []
        for tile_type in ['wan', 'tiao', 'tong']:
            for value in range(1, 10):
                for _ in range(4):
                    wall.append(TileUtils.create_tile(tile_type, value))
        
        for tile_type in ['feng']:
            for value in range(1, 5):
                for _ in range(4):
                    wall.append(TileUtils.create_tile(tile_type, value))
        
        for tile_type in ['jian']:
            for value in range(1, 4):
                for _ in range(4):
                    wall.append(TileUtils.create_tile(tile_type, value))
        
        return wall
    
    @staticmethod
    def shuffle_wall(wall: List[Dict], seed: Optional[int] = None) -> List[Dict]:
        if seed is not None:
            np.random.seed(seed)
        shuffled = wall.copy()
        np.random.shuffle(shuffled)
        return shuffled
    
    @staticmethod
    def sort_hand(hand: List[Dict]) -> List[Dict]:
        return sorted(hand, key=lambda x: x['id'])
    
    @staticmethod
    def count_tiles(tiles: List[Dict]) -> Dict[int, int]:
        counts = {}
        for tile in tiles:
            tile_id = tile['id']
            counts[tile_id] = counts.get(tile_id, 0) + 1
        return counts
    
    @staticmethod
    def tiles_to_array(tiles: List[Dict], num_tile_types: int = 34) -> np.ndarray:
        arr = np.zeros(num_tile_types, dtype=np.int8)
        for tile in tiles:
            arr[tile['id']] += 1
        return arr
    
    @staticmethod
    def is_terminal_tile(tile: Dict) -> bool:
        tile_type = tile['type']
        value = tile['value']
        if tile_type in ['wan', 'tiao', 'tong']:
            return value == 1 or value == 9
        return True
    
    @staticmethod
    def is_honor_tile(tile: Dict) -> bool:
        return tile['type'] in ['feng', 'jian']
    
    @staticmethod
    def get_next_tile(tile: Dict) -> Optional[Dict]:
        tile_type = tile['type']
        value = tile['value']
        if tile_type in ['wan', 'tiao', 'tong'] and value < 9:
            return TileUtils.create_tile(tile_type, value + 1)
        return None
    
    @staticmethod
    def get_prev_tile(tile: Dict) -> Optional[Dict]:
        tile_type = tile['type']
        value = tile['value']
        if tile_type in ['wan', 'tiao', 'tong'] and value > 1:
            return TileUtils.create_tile(tile_type, value - 1)
        return None
