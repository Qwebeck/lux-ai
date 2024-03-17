from ray.rllib.models.modelv2 import restore_original_dimensions
from typing import Optional
from luxai_s2.actions import move_deltas
from luxai_s2.env import LuxAI_S2
from ray.rllib.models.modelv2 import restore_original_dimensions
import gymnasium as gym
import tree
import numpy as np
import torch
from impl_config import ActDims, EnvParam, FactoryActType, ModelParam, UnitActType
from lux.config import EnvConfig
from lux.kit import Board, GameState, obs_to_game_state
from lux.utils import my_turn_to_place_factory
from policy.early_setup_player import EarlySetupPlayer
from policy.net import Net

import numpy as np
from enum import Enum
import dataclasses


class Agent:
    def __init__(self, player_id: str, env_config: EnvConfig) -> None:
        self.player_id = player_id
        self.opponent = "player_1" if player_id == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg = env_config
        self.net = Net()

    def act(self, obs, remainingOverageTime: int = 60):
        obs = obs['obs']
        player_obs, valid_actions = obs['my_obs'], obs['valid_actions']

        global_feature, map_feature, action_feature = player_obs.global_features, player_obs.map_feature, player_obs.action_feature

        logp, logits, value, action, entropy = self.net.forward(
            global_feature,
            map_feature,
            tree.map_structure(lambda x:
                               x, action_feature),
            tree.map_structure(lambda x:
                               x,
                               valid_actions)
        )
        return logp, logits, value, action, entropy
