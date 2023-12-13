import tree
import numpy as np
import torch
from helpers.action_parser import ActionParser
from helpers.feature_parser import FeatureParser
from lux.config import EnvConfig
from lux.kit import obs_to_game_state
from policy.early_setup_player import Player
from policy.net import Net


class Agent:
    def __init__(self, player: str, env_config: EnvConfig) -> None:
        self.player = player
        self.opponent = "player_1" if player == "player_0" else "player_0"
        np.random.seed(env_config.seed)
        self.env_cfg = env_config
        self.feature_parser = FeatureParser()
        self.action_parser = ActionParser()  # TODO
        self.early_setup_player = Player()  # TODO
        self.net = Net()  # TODO

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if self.early_setup_player:
            return self.early_setup_player.act(step, obs, remainingOverageTime)
        else:
            return self.act(step, obs, remainingOverageTime)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        game_state = obs_to_game_state(step, self.env_cfg, obs[self.player])
        action_queues = []
        for units in game_state.units.values():
            for u in units:
                action_queues.append(u.action_queue)
        features = self.feature_parser._get_map_features(obs, self.player)
        valid_actions = self.action_parser.get_valid_actions(
            game_state, self.player)

        def np2torch(x, dtype): return torch.from_numpy(x).type(dtype)

        logp, value, action, entropy = self.net.forward(
            np2torch(features.global_features, torch.float32),
            np2torch(features.map_feature, torch.float32),
            tree.map_structure(lambda x: np2torch(
                x, torch.int16), features.action_feature),
            tree.map_structure(lambda x: np2torch(
                x, torch.bool), valid_actions)
        )
        action = tree.map_structure(lambda x: x.detach().numpy()[0], action)
        action = self.action_parser._parse(game_state, self.player, action)
        return action

        # features = self.feature_parser.parse(obs, self.env_cfg)[self.player]
