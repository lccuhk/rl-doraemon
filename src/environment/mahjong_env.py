import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
from ..utils.tile_utils import TileUtils
from .rules import MahjongRules

class ActionType(Enum):
    DISCARD_BASE = 0
    CHI = 34
    PON = 35
    KAN = 36
    RIICHI = 37
    RON = 38
    TSUMO = 39
    PASS = 40

class GamePhase(Enum):
    DEALING = 0
    DRAWING = 1
    DISCARDING = 2
    RESPONDING = 3
    GAME_OVER = 4

class MahjongEnv:
    def __init__(self, num_players: int = 4, seed: Optional[int] = None, 
                 use_intermediate_rewards: bool = True):
        self.num_players = num_players
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.use_intermediate_rewards = use_intermediate_rewards
        
        self.logger = logging.getLogger('mahjong_env')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.wall: List[Dict] = []
        self.dora_indicators: List[Dict] = []
        self.players: List[Dict] = []
        self.current_player: int = 0
        self.dealer: int = 0
        self.round: int = 0
        self.honba: int = 0
        self.phase: GamePhase = GamePhase.DEALING
        self.last_discard: Optional[Dict] = None
        self.last_discarder: int = -1
        self.game_over: bool = False
        self.winner: int = -1
        self.scores: List[int] = [25000] * num_players
        self.discards: List[List[Dict]] = [[] for _ in range(num_players)]
        self.melds: List[List[List[Dict]]] = [[] for _ in range(num_players)]
        self.is_menzen: List[bool] = [True] * num_players
        self.is_riichi: List[bool] = [False] * num_players
        self.riichi_sticks: int = 0
        
        self.shanten_history: List[int] = [8] * num_players
        
        self.action_space_size = 41
        self.observation_shape = (6, 34, 4)
        
        self.reset()
    
    def reset(self) -> Dict:
        self.wall = TileUtils.create_full_wall()
        self.wall = TileUtils.shuffle_wall(self.wall, self.seed)
        
        self.players = []
        for i in range(self.num_players):
            hand = self.wall[:13]
            self.wall = self.wall[13:]
            hand = TileUtils.sort_hand(hand)
            self.players.append({
                'hand': hand,
                'score': 25000,
                'is_riichi': False,
                'is_menzen': True,
                'melds': [],
                'discards': []
            })
        
        dead_wall = self.wall[-14:]
        self.wall = self.wall[:-14]
        self.dora_indicators = [dead_wall[0]]
        
        self.current_player = self.dealer
        self.last_discard = None
        self.last_discarder = -1
        self.game_over = False
        self.winner = -1
        self.discards = [[] for _ in range(self.num_players)]
        self.melds = [[] for _ in range(self.num_players)]
        self.is_menzen = [True] * self.num_players
        self.is_riichi = [False] * self.num_players
        self.riichi_sticks = 0
        
        if self.wall:
            new_tile = self.wall.pop(0)
            self.players[self.current_player]['hand'].append(new_tile)
            self.players[self.current_player]['hand'] = TileUtils.sort_hand(
                self.players[self.current_player]['hand']
            )
        
        self.phase = GamePhase.DRAWING
        
        return self._get_observation(self.current_player)
    
    def _get_observation(self, player_id: int) -> Dict:
        player = self.players[player_id]
        
        hand_rep = np.zeros((34, 4), dtype=np.float32)
        for tile in player['hand']:
            tile_id = tile['id']
            for i in range(4):
                if hand_rep[tile_id][i] == 0:
                    hand_rep[tile_id][i] = 1
                    break
        
        table_rep = np.zeros((34, 4), dtype=np.float32)
        for p_id in range(self.num_players):
            for tile in self.discards[p_id]:
                tile_id = tile['id']
                for i in range(4):
                    if table_rep[tile_id][i] == 0:
                        table_rep[tile_id][i] = 1
                        break
        
        melds_rep = np.zeros((34, 4), dtype=np.float32)
        for p_id in range(self.num_players):
            for meld in self.melds[p_id]:
                for tile in meld:
                    tile_id = tile['id']
                    for i in range(4):
                        if melds_rep[tile_id][i] == 0:
                            melds_rep[tile_id][i] = 1
                            break
        
        dora_rep = np.zeros((34, 4), dtype=np.float32)
        for tile in self.dora_indicators:
            tile_id = tile['id']
            dora_rep[tile_id][0] = 1
        
        player_info = np.zeros((34, 4), dtype=np.float32)
        player_info[player_id][0] = 1
        player_info[self.dealer][1] = 1
        if self.is_riichi[player_id]:
            player_info[player_id][2] = 1
        
        game_info = np.zeros((34, 4), dtype=np.float32)
        game_info[0][0] = self.round / 8
        game_info[0][1] = self.honba / 4
        game_info[0][2] = len(self.wall) / 70
        
        obs = np.stack([hand_rep, table_rep, melds_rep, dora_rep, player_info, game_info])
        
        legal_actions = self._get_legal_actions(player_id)
        
        return {
            'obs': obs,
            'legal_actions': legal_actions,
            'player_id': player_id,
            'current_player': self.current_player,
            'phase': self.phase.value,
            'scores': [p['score'] for p in self.players],
            'hand': player['hand'],
            'discards': self.discards.copy(),
            'melds': self.melds.copy(),
            'is_riichi': self.is_riichi.copy(),
            'dora_indicators': self.dora_indicators,
            'wall_remaining': len(self.wall)
        }
    
    def _get_legal_actions(self, player_id: int) -> List[int]:
        legal_actions = []
        player = self.players[player_id]
        
        if self.phase == GamePhase.DRAWING:
            if player_id == self.current_player:
                is_win, _ = MahjongRules.check_tsumo(
                    player['hand'], 
                    self.melds[player_id],
                    self.is_menzen[player_id],
                    self.is_riichi[player_id]
                )
                if is_win:
                    legal_actions.append(ActionType.TSUMO.value)
                
                if self.is_menzen[player_id] and not self.is_riichi[player_id]:
                    legal_actions.append(ActionType.RIICHI.value)
                
                kans = MahjongRules.check_kan(player['hand'])
                if kans:
                    legal_actions.append(ActionType.KAN.value)
                
                legal_actions.append(ActionType.PASS.value)
        
        elif self.phase == GamePhase.DISCARDING:
            if player_id == self.current_player:
                for i in range(len(player['hand'])):
                    legal_actions.append(i)
        
        elif self.phase == GamePhase.RESPONDING:
            if player_id != self.last_discarder and self.last_discard is not None:
                is_ron, _ = MahjongRules.check_ron(
                    player['hand'],
                    self.last_discard,
                    self.melds[player_id],
                    self.is_menzen[player_id],
                    self.is_riichi[player_id]
                )
                if is_ron:
                    legal_actions.append(ActionType.RON.value)
                
                if MahjongRules.check_pon(player['hand'], self.last_discard):
                    legal_actions.append(ActionType.PON.value)
                
                chis = MahjongRules.check_chi(player['hand'], self.last_discard)
                if chis:
                    legal_actions.append(ActionType.CHI.value)
                
                if MahjongRules.check_kan(player['hand'], self.last_discard):
                    legal_actions.append(ActionType.KAN.value)
                
                legal_actions.append(ActionType.PASS.value)
        
        return legal_actions
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        if self.game_over:
            return self._get_observation(self.current_player), 0.0, True, {'game_over': True}
        
        player = self.players[self.current_player]
        reward = 0.0
        info = {}
        
        old_shanten = self._calculate_shanten(self.current_player)
        
        if self.phase == GamePhase.DRAWING:
            if action == ActionType.TSUMO.value:
                is_win, win_info = MahjongRules.check_tsumo(
                    player['hand'],
                    self.melds[self.current_player],
                    self.is_menzen[self.current_player],
                    self.is_riichi[self.current_player]
                )
                if is_win:
                    self._handle_tsumo(win_info)
                    info['win_type'] = 'tsumo'
                    info['win_info'] = win_info
            elif action == ActionType.RIICHI.value:
                self.is_riichi[self.current_player] = True
                player['score'] -= 1000
                self.riichi_sticks += 1
                self.phase = GamePhase.DISCARDING
                
                if self.use_intermediate_rewards:
                    reward += 0.15
                    info['riichi_reward'] = 0.15
            elif action == ActionType.KAN.value:
                kans = MahjongRules.check_kan(player['hand'])
                if kans:
                    self._handle_kan(kans[0], is_closed=True)
                    
                    if self.use_intermediate_rewards:
                        reward += 0.1
                        info['kan_reward'] = 0.1
            else:
                self.phase = GamePhase.DISCARDING
        
        elif self.phase == GamePhase.DISCARDING:
            if 0 <= action < len(player['hand']):
                discarded_tile = player['hand'].pop(action)
                self.discards[self.current_player].append(discarded_tile)
                self.last_discard = discarded_tile
                self.last_discarder = self.current_player
                self.phase = GamePhase.RESPONDING
                info['discarded'] = discarded_tile
                
                if self.use_intermediate_rewards:
                    discard_reward = self._calculate_discard_reward(
                        self.current_player, discarded_tile
                    )
                    reward += discard_reward
                    info['discard_reward'] = discard_reward
                
                self._process_responses()
        
        elif self.phase == GamePhase.RESPONDING:
            if action == ActionType.RON.value:
                is_win, win_info = MahjongRules.check_ron(
                    player['hand'],
                    self.last_discard,
                    self.melds[self.current_player],
                    self.is_menzen[self.current_player],
                    self.is_riichi[self.current_player]
                )
                if is_win:
                    self._handle_ron(win_info)
                    info['win_type'] = 'ron'
                    info['win_info'] = win_info
            elif action == ActionType.PON.value:
                self._handle_pon()
                
                if self.use_intermediate_rewards:
                    reward += 0.08
                    info['pon_reward'] = 0.08
            elif action == ActionType.CHI.value:
                self._handle_chi()
                
                if self.use_intermediate_rewards:
                    reward += 0.05
                    info['chi_reward'] = 0.05
            elif action == ActionType.KAN.value:
                self._handle_kan(self.last_discard, is_closed=False)
                
                if self.use_intermediate_rewards:
                    reward += 0.1
                    info['kan_reward'] = 0.1
            elif action == ActionType.PASS.value:
                self._next_turn()
        
        if self.use_intermediate_rewards and not self.game_over:
            new_shanten = self._calculate_shanten(self.current_player)
            shanten_improvement = old_shanten - new_shanten
            
            if shanten_improvement > 0:
                shanten_reward = shanten_improvement * 0.12
                reward += shanten_reward
                info['shanten_reward'] = shanten_reward
            elif shanten_improvement < 0:
                reward -= 0.05
                info['shanten_penalty'] = -0.05
            
            if new_shanten == 0:
                reward += 0.25
                info['tenpai_reward'] = 0.25
            
            self.shanten_history[self.current_player] = new_shanten
        
        if self.game_over:
            final_rewards = self._calculate_final_rewards()
            return self._get_observation(self.current_player), final_rewards[self.current_player], True, info
        
        return self._get_observation(self.current_player), reward, False, info
    
    def _calculate_shanten(self, player_id: int) -> int:
        player = self.players[player_id]
        return MahjongRules.get_shanten(player['hand'], self.melds[player_id])
    
    def _calculate_discard_reward(self, player_id: int, discarded_tile: Dict) -> float:
        player = self.players[player_id]
        reward = 0.0
        
        if TileUtils.is_honor_tile(discarded_tile):
            hand_counts = TileUtils.count_tiles(player['hand'])
            if hand_counts.get(discarded_tile['id'], 0) == 0:
                reward += 0.02
        
        if TileUtils.is_terminal_tile(discarded_tile):
            hand_counts = TileUtils.count_tiles(player['hand'])
            if hand_counts.get(discarded_tile['id'], 0) == 0:
                reward += 0.01
        
        return reward
    
    def _process_responses(self):
        for i in range(1, self.num_players):
            responder_id = (self.last_discarder + i) % self.num_players
            self.current_player = responder_id
            
            legal_actions = self._get_legal_actions(responder_id)
            
            if ActionType.RON.value in legal_actions:
                self.phase = GamePhase.RESPONDING
                return
            elif ActionType.PON.value in legal_actions:
                self.phase = GamePhase.RESPONDING
                return
            elif ActionType.CHI.value in legal_actions and responder_id == (self.last_discarder + 1) % self.num_players:
                self.phase = GamePhase.RESPONDING
                return
        
        self._next_turn()
    
    def _handle_tsumo(self, win_info: Dict):
        self.game_over = True
        self.winner = self.current_player
        
        player = self.players[self.current_player]
        hand_symbols = [TileUtils.get_tile_symbol(t) for t in player['hand']]
        meld_symbols = []
        for meld in self.melds[self.current_player]:
            meld_symbols.append([TileUtils.get_tile_symbol(t) for t in meld])
        
        self.logger.warning("="*70)
        self.logger.warning(f"[自摸和牌! 玩家: {self.current_player}")
        self.logger.warning(f"  手牌: {hand_symbols}")
        if meld_symbols:
            self.logger.warning(f"  副露: {meld_symbols}")
        self.logger.warning(f"  役: {win_info.get('yaku', [])}")
        self.logger.warning(f"  番数: {win_info.get('han', 0)}")
        self.logger.warning(f"  符数: {win_info.get('fu', 0)}")
        self.logger.warning(f"  门清: {self.is_menzen[self.current_player]}")
        self.logger.warning(f"  立直: {self.is_riichi[self.current_player]}")
        self.logger.warning("="*70)
        
        score_info = MahjongRules.calculate_score(
            win_info.get('han', 1),
            win_info.get('fu', 30),
            is_tsumo=True,
            is_dealer=(self.current_player == self.dealer),
            is_yakuman=win_info.get('is_yakuman', False)
        )
        
        winner = self.players[self.current_player]
        winner['score'] += self.riichi_sticks * 1000
        
        if self.current_player == self.dealer:
            payment = score_info['each_payment']
            for i in range(self.num_players):
                if i != self.current_player:
                    self.players[i]['score'] -= payment
                    winner['score'] += payment
        else:
            dealer_payment = score_info['each_payment']['dealer']
            other_payment = score_info['each_payment']['others']
            for i in range(self.num_players):
                if i == self.dealer:
                    self.players[i]['score'] -= dealer_payment
                    winner['score'] += dealer_payment
                elif i != self.current_player:
                    self.players[i]['score'] -= other_payment
                    winner['score'] += other_payment
    
    def _handle_ron(self, win_info: Dict):
        self.game_over = True
        self.winner = self.current_player
        
        player = self.players[self.current_player]
        hand_symbols = [TileUtils.get_tile_symbol(t) for t in player['hand']]
        meld_symbols = []
        for meld in self.melds[self.current_player]:
            meld_symbols.append([TileUtils.get_tile_symbol(t) for t in meld])
        
        self.logger.warning("="*70)
        self.logger.warning(f"[荣和和牌! 玩家: {self.current_player}")
        self.logger.warning(f"  手牌: {hand_symbols}")
        if meld_symbols:
            self.logger.warning(f"  副露: {meld_symbols}")
        self.logger.warning(f"  和牌: {TileUtils.get_tile_symbol(self.last_discard)} (来自玩家{self.last_discarder})")
        self.logger.warning(f"  役: {win_info.get('yaku', [])}")
        self.logger.warning(f"  番数: {win_info.get('han', 0)}")
        self.logger.warning(f"  符数: {win_info.get('fu', 0)}")
        self.logger.warning(f"  门清: {self.is_menzen[self.current_player]}")
        self.logger.warning(f"  立直: {self.is_riichi[self.current_player]}")
        self.logger.warning("="*70)
        
        score_info = MahjongRules.calculate_score(
            win_info.get('han', 1),
            win_info.get('fu', 30),
            is_tsumo=False,
            is_dealer=(self.current_player == self.dealer),
            is_yakuman=win_info.get('is_yakuman', False)
        )
        
        winner = self.players[self.current_player]
        discarder = self.players[self.last_discarder]
        
        winner['score'] += self.riichi_sticks * 1000
        payment = score_info['each_payment']
        discarder['score'] -= payment
        winner['score'] += payment
    
    def _handle_pon(self):
        player = self.players[self.current_player]
        tile = self.last_discard
        
        removed = 0
        for i, t in enumerate(player['hand']):
            if t['id'] == tile['id'] and removed < 2:
                player['hand'].pop(i)
                removed += 1
                if removed == 2:
                    break
        
        pon_meld = [tile, tile, tile]
        self.melds[self.current_player].append(pon_meld)
        self.is_menzen[self.current_player] = False
        
        self.discards[self.last_discarder].pop()
        self.phase = GamePhase.DISCARDING
    
    def _handle_chi(self):
        player = self.players[self.current_player]
        tile = self.last_discard
        
        chis = MahjongRules.check_chi(player['hand'], tile)
        if chis:
            chi_meld = chis[0]
            for t in chi_meld:
                if t['id'] != tile['id']:
                    for i, ht in enumerate(player['hand']):
                        if ht['id'] == t['id']:
                            player['hand'].pop(i)
                            break
            
            self.melds[self.current_player].append(chi_meld)
            self.is_menzen[self.current_player] = False
            
            self.discards[self.last_discarder].pop()
            self.phase = GamePhase.DISCARDING
    
    def _handle_kan(self, tile: Dict, is_closed: bool):
        player = self.players[self.current_player]
        
        if is_closed:
            removed = 0
            for i, t in enumerate(player['hand']):
                if t['id'] == tile['id'] and removed < 4:
                    player['hand'].pop(i)
                    removed += 1
                    if removed == 4:
                        break
        else:
            removed = 0
            for i, t in enumerate(player['hand']):
                if t['id'] == tile['id'] and removed < 3:
                    player['hand'].pop(i)
                    removed += 1
                    if removed == 3:
                        break
            self.discards[self.last_discarder].pop()
        
        kan_meld = [tile] * 4
        self.melds[self.current_player].append(kan_meld)
        if not is_closed:
            self.is_menzen[self.current_player] = False
        
        if self.wall:
            new_tile = self.wall.pop(0)
            player['hand'].append(new_tile)
            player['hand'] = TileUtils.sort_hand(player['hand'])
        
        self.phase = GamePhase.DRAWING
    
    def _next_turn(self):
        self.current_player = (self.current_player + 1) % self.num_players
        
        if self.wall:
            new_tile = self.wall.pop(0)
            self.players[self.current_player]['hand'].append(new_tile)
            self.players[self.current_player]['hand'] = TileUtils.sort_hand(
                self.players[self.current_player]['hand']
            )
            self.phase = GamePhase.DRAWING
        else:
            self.game_over = True
            self.winner = -1
    
    def _calculate_final_rewards(self) -> List[float]:
        rewards = [0.0] * self.num_players
        
        if self.winner >= 0:
            for i in range(self.num_players):
                if i == self.winner:
                    rewards[i] = 15.0
                else:
                    rewards[i] = -5.0
        else:
            scores = [p['score'] for p in self.players]
            max_score = max(scores)
            for i in range(self.num_players):
                if scores[i] == max_score:
                    rewards[i] = 0.3
                else:
                    rewards[i] = -1.0
        
        return rewards
    
    def render(self, player_id: int = 0):
        player = self.players[player_id]
        print(f"\n{'='*60}")
        print(f"玩家 {player_id} 的视角")
        print(f"{'='*60}")
        
        print(f"\n手牌:")
        hand_symbols = [TileUtils.get_tile_symbol(t) for t in player['hand']]
        hand_names = [TileUtils.get_tile_name(t) for t in player['hand']]
        print(f"  符号: {' '.join(hand_symbols)}")
        print(f"  名称: {' '.join(hand_names)}")
        
        if self.melds[player_id]:
            print(f"\n副露:")
            for i, meld in enumerate(self.melds[player_id]):
                meld_symbols = [TileUtils.get_tile_symbol(t) for t in meld]
                print(f"  副露{i+1}: {' '.join(meld_symbols)}")
        
        print(f"\n宝牌指示牌:")
        for tile in self.dora_indicators:
            print(f"  {TileUtils.get_tile_symbol(tile)} {TileUtils.get_tile_name(tile)}")
        
        print(f"\n舍牌:")
        for i in range(self.num_players):
            if self.discards[i]:
                discard_symbols = [TileUtils.get_tile_symbol(t) for t in self.discards[i]]
                print(f"  玩家{i}: {' '.join(discard_symbols)}")
        
        print(f"\n分数:")
        for i in range(self.num_players):
            status = []
            if i == self.dealer:
                status.append("庄家")
            if self.is_riichi[i]:
                status.append("立直")
            status_str = f" ({', '.join(status)})" if status else ""
            print(f"  玩家{i}: {self.players[i]['score']}点{status_str}")
        
        print(f"\n剩余牌数: {len(self.wall)}")
        print(f"当前阶段: {self.phase.name}")
        if self.last_discard:
            print(f"最后舍牌: {TileUtils.get_tile_symbol(self.last_discard)} {TileUtils.get_tile_name(self.last_discard)}")
        print(f"{'='*60}\n")
    
    def get_state(self) -> Dict:
        return {
            'players': self.players,
            'wall': self.wall,
            'dora_indicators': self.dora_indicators,
            'current_player': self.current_player,
            'dealer': self.dealer,
            'round': self.round,
            'honba': self.honba,
            'phase': self.phase,
            'last_discard': self.last_discard,
            'last_discarder': self.last_discarder,
            'game_over': self.game_over,
            'winner': self.winner,
            'discards': self.discards,
            'melds': self.melds,
            'is_menzen': self.is_menzen,
            'is_riichi': self.is_riichi,
            'riichi_sticks': self.riichi_sticks
        }
    
    def set_state(self, state: Dict):
        self.players = state['players']
        self.wall = state['wall']
        self.dora_indicators = state['dora_indicators']
        self.current_player = state['current_player']
        self.dealer = state['dealer']
        self.round = state['round']
        self.honba = state['honba']
        self.phase = state['phase']
        self.last_discard = state['last_discard']
        self.last_discarder = state['last_discarder']
        self.game_over = state['game_over']
        self.winner = state['winner']
        self.discards = state['discards']
        self.melds = state['melds']
        self.is_menzen = state['is_menzen']
        self.is_riichi = state['is_riichi']
        self.riichi_sticks = state['riichi_sticks']
