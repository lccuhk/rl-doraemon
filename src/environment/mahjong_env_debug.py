import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import logging
from ..utils.tile_utils import TileUtils
from .rules import MahjongRules
from .mahjong_env import MahjongEnv, ActionType, GamePhase

class MahjongEnvDebug(MahjongEnv):
    def __init__(self, num_players: int = 4, seed: Optional[int] = None, 
                 use_intermediate_rewards: bool = True, debug_level: int = 1):
        self.debug_level = debug_level
        self.logger = logging.getLogger('mahjong_env_debug')
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.step_count = 0
        self.episode_count = 0
        self.win_detected_count = 0
        self.win_opportunities = {
            'tsumo_available': 0,
            'ron_available': 0,
            'tsumo_taken': 0,
            'ron_taken': 0
        }
        
        super().__init__(num_players, seed, use_intermediate_rewards)
    
    def reset(self) -> Dict:
        self.step_count = 0
        self.episode_count += 1
        self.win_opportunities = {
            'tsumo_available': 0,
            'ron_available': 0,
            'tsumo_taken': 0,
            'ron_taken': 0
        }
        
        if self.debug_level >= 1:
            self.logger.info(f"=== 第 {self.episode_count} 回合开始 ===")
        
        return super().reset()
    
    def _get_legal_actions(self, player_id: int) -> List[int]:
        legal_actions = super()._get_legal_actions(player_id)
        
        if self.debug_level >= 2:
            phase_name = self.phase.name
            has_tsumo = ActionType.TSUMO.value in legal_actions
            has_ron = ActionType.RON.value in legal_actions
            has_riichi = ActionType.RIICHI.value in legal_actions
            has_pon = ActionType.PON.value in legal_actions
            has_chi = ActionType.CHI.value in legal_actions
            has_kan = ActionType.KAN.value in legal_actions
            
            if has_tsumo:
                self.win_opportunities['tsumo_available'] += 1
                if self.debug_level >= 1:
                    self.logger.warning(f"[自摸机会] 玩家{player_id} 在 {phase_name} 阶段有自摸机会！")
            
            if has_ron:
                self.win_opportunities['ron_available'] += 1
                if self.debug_level >= 1:
                    self.logger.warning(f"[荣和机会] 玩家{player_id} 在 {phase_name} 阶段有荣和机会！")
            
            if has_riichi or has_pon or has_chi or has_kan:
                self.logger.debug(f"[合法动作] 玩家{player_id} 阶段:{phase_name} "
                                f"立直:{has_riichi} 碰:{has_pon} 吃:{has_chi} 杠:{has_kan}")
        
        return legal_actions
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        self.step_count += 1
        player = self.players[self.current_player]
        current_player = self.current_player
        phase_name = self.phase.name
        
        if self.debug_level >= 2:
            self.logger.debug(f"[步骤{self.step_count}] 玩家{current_player} 阶段:{phase_name} 动作:{action}")
        
        if self.phase == GamePhase.DRAWING:
            if action == ActionType.TSUMO.value:
                is_win, win_info = MahjongRules.check_tsumo(
                    player['hand'],
                    self.melds[self.current_player],
                    self.is_menzen[self.current_player]
                )
                if is_win:
                    self.win_opportunities['tsumo_taken'] += 1
                    self.win_detected_count += 1
                    if self.debug_level >= 1:
                        self.logger.warning(f"[自摸成功] 玩家{current_player} 选择自摸！"
                                          f"番数:{win_info.get('han', 0)} 符数:{win_info.get('fu', 0)}")
                else:
                    if self.debug_level >= 1:
                        self.logger.error(f"[自摸失败] 玩家{current_player} 选择自摸但检测失败！")
            elif ActionType.TSUMO.value in self._get_legal_actions(current_player):
                if self.debug_level >= 1:
                    self.logger.warning(f"[放弃自摸] 玩家{current_player} 有自摸机会但选择了动作:{action}")
        
        elif self.phase == GamePhase.RESPONDING:
            if action == ActionType.RON.value:
                is_win, win_info = MahjongRules.check_ron(
                    player['hand'],
                    self.last_discard,
                    self.melds[self.current_player],
                    self.is_menzen[self.current_player]
                )
                if is_win:
                    self.win_opportunities['ron_taken'] += 1
                    self.win_detected_count += 1
                    if self.debug_level >= 1:
                        self.logger.warning(f"[荣和成功] 玩家{current_player} 选择荣和！"
                                          f"番数:{win_info.get('han', 0)} 符数:{win_info.get('fu', 0)}")
            elif ActionType.RON.value in self._get_legal_actions(current_player):
                if self.debug_level >= 1:
                    self.logger.warning(f"[放弃荣和] 玩家{current_player} 有荣和机会但选择了动作:{action}")
        
        result = super().step(action)
        next_obs, reward, done, info = result
        
        if self.debug_level >= 2 and reward != 0:
            reward_details = []
            for key, value in info.items():
                if 'reward' in key.lower() or 'penalty' in key.lower():
                    reward_details.append(f"{key}:{value}")
            if reward_details:
                self.logger.debug(f"[奖励] 玩家{current_player} 总奖励:{reward} 详情:{', '.join(reward_details)}")
        
        if done:
            self._log_episode_summary()
        
        return result
    
    def _log_episode_summary(self):
        if self.winner >= 0:
            if self.debug_level >= 1:
                self.logger.info(f"=== 回合结束: 玩家{self.winner} 和牌！ ===")
                self.logger.info(f"  总步数: {self.step_count}")
                self.logger.info(f"  自摸机会: {self.win_opportunities['tsumo_available']}, 自摸成功: {self.win_opportunities['tsumo_taken']}")
                self.logger.info(f"  荣和机会: {self.win_opportunities['ron_available']}, 荣和成功: {self.win_opportunities['ron_taken']}")
        else:
            if self.debug_level >= 1:
                self.logger.info(f"=== 回合结束: 流局 ===")
                self.logger.info(f"  总步数: {self.step_count}")
                self.logger.info(f"  自摸机会: {self.win_opportunities['tsumo_available']}, 自摸成功: {self.win_opportunities['tsumo_taken']}")
                self.logger.info(f"  荣和机会: {self.win_opportunities['ron_available']}, 荣和成功: {self.win_opportunities['ron_taken']}")
                if self.win_opportunities['tsumo_available'] > 0 or self.win_opportunities['ron_available'] > 0:
                    self.logger.warning(f"  [问题] 本回合有和牌机会但最终流局！")
    
    def get_debug_stats(self) -> Dict:
        return {
            'episode_count': self.episode_count,
            'win_detected_count': self.win_detected_count,
            'win_opportunities': self.win_opportunities.copy()
        }
