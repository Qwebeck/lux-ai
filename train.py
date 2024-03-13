# Find some really good tutorial and implement single agent learning
# https://docs.ray.io/en/latest/ray-core/examples/plot_pong_example.html
from impl_config import EnvParam
from luxai_s2_patch import install_patch
from functools import partial
from helpers.feature_parser import LuxFeature
import lux.kit
import torch
from typing import Any
from luxai_s2.state.state import ObservationStateDict
from luxai_s2.env import LuxAI_S2
import gymnasium as gym
import gymnasium.spaces
import numpy as np
from agent import Agent
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory


from lux.unit import Unit
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print

from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModuleConfig
from ray.rllib.algorithms import PPO
from ray.rllib.utils.nested_dict import NestedDict
from typing import Mapping
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from policy.spaces import get_observation_space, get_action_space


MY_PLAYER = 'player_0'

# TODO
# Based on available code:
# - find how they handle obs space
# - run training


""""
Build custom obs space (steal from the baseline)
Pass it to the model in some way (run run ru n)
Run the training!!!. 



How to access 
"""


class AgentModule(TorchRLModule):
    def __init__(self, config: RLModuleConfig, agent: Agent) -> None:
        super().__init__(config)
        self.agent = agent
        self._parameters = self.agent.net.state_dict()

    # @override(TorchRLModule)
    # TODO: hack
    def set_state(self, state_dict: Mapping[str, Any]) -> None:
        self.load_state_dict(state_dict, False)

    def _forward_inference(self, batch: NestedDict, **kwargs):
        with torch.no_grad():
            return self._forward_train(batch, **kwargs)

    def _forward_exploration(self, batch: NestedDict, **kwargs):
        with torch.no_grad():
            return self._forward_train(batch, **kwargs)

    def _forward_train(self, batch: NestedDict, **kwargs) -> Mapping[str, Any]:
        actions = self.agent.act(0, batch)
        return {
            "actions": actions
        }


class RayWrapper(gym.Env):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.lux_env = LuxAI_S2(*args, **kwargs)
        self.player_id = "player_0"
        self.opponent_id = "player_1"
        self.agent = Agent(self.player_id, self.lux_env.env_cfg, self.lux_env)

    def reset(self):
        obs, _ = self.lux_env.reset()
        game_state = lux.kit.obs_to_game_state(0, self.lux_env.env_cfg, obs)
        while self.lux_env.state.real_env_steps < 0:
            my_action = self.agent.early_setup(
                game_state.env_steps, obs[self.player_id])
            obs, *other = self.lux_env.step({
                self.player_id: my_action,
                self.opponent_id: dict()
            })
            game_state = lux.kit.obs_to_game_state(
                game_state.env_steps, self.lux_env.env_cfg, obs)

        return (obs, *other)

    def step(self, action: Any) -> Any:
        obs, *other = self.lux_env.step(
            {'player_0': dict(), 'player_1': dict()})
        return (obs, *other)

    def render(self, mode: str = 'human') -> Any:
        return self.lux_env.render(mode)


def train():
    env = RayWrapper(verbose=3, collect_stats=True,
                     FACTORY_WATER_CONSUMPTION=0)
    module_spec = SingleAgentRLModuleSpec(
        module_class=partial(
            AgentModule, agent=env.agent),
        observation_space=env.observation_space,
        action_space=env.action_space,
    )

    # spec.build().parameters()
    config = (PPOConfig()
              .experimental(_enable_new_api_stack=True, _disable_preprocessor_api=True, _disable_initialize_loss_from_dummy_batch=True)
              .environment(env=RayWrapper,
                           disable_env_checking=True,
                           env_config={
                               "verbose": 0,
                               "collect_stats": True,
                               "FACTORY_WATER_CONSUMPTION": 0,
                               "max_episode_steps": 1000,
                               "BIDDING_SYSTEM": False})
              .rl_module(
                  rl_module_spec=module_spec,
    )
        .rollouts(num_rollout_workers=0)

    )

    algo = config.build()

    for _ in range(10):
        result = algo.train()
        print(pretty_print(result))


if __name__ == "__main__":
    # install_patch()
    train()
