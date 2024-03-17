# Find some really good tutorial and implement single agent learning
# https://docs.ray.io/en/latest/ray-core/examples/plot_pong_example.html
from lux.kit import obs_to_game_state
from collections import defaultdict
from dataclasses import asdict
from math import prod
import tree
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.models.torch.torch_distributions import TorchDeterministic, TorchMultiDistribution, TorchDiagGaussian, TorchCategorical

from helpers.get_valid_actions import get_valid_actions
from impl_config import EnvParam, ModelParam
from luxai_s2_patch import install_patch
from functools import partial
from parsers.action_parser import ActionParser
from parsers.feature_parser import FeatureParser
import lux.kit
import torch
from typing import Any
from luxai_s2.state.state import ObservationStateDict
from luxai_s2.env import LuxAI_S2
from lux.config import EnvConfig
import gymnasium as gym
import numpy as np
from agent import Agent


from ray.rllib.core.rl_module import RLModule
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
from parsers.feature_parser import FeatureParser
from policy.early_setup_player import EarlySetupPlayer
from policy.spaces import get_observation_space, get_action_space


MY_PLAYER = 'player_0'

"""
I have some progress, but I can't say the model is training. 
I also don't know how to make at multiagent and how to deploy it somewhere.
---> I should have it fully working till the end of the next week

---
I should look for a job and therefore, should perpare my CV and apply to interesting positions. I still underprepared for positions I want to apply. 

---
I have a message from Lazarz and I don't know what he wants from me.

---
I want to contribute to the drones project

"""


class CustomDistr(TorchCategorical):
    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
    ) -> "TorchCategorical":
        """Creates this Distribution from logits (and additional arguments).

        If you wish to create this distribution from logits only, please refer to
        `Distribution.get_partial_dist_cls()`.

        Args:
            logits: The tensor containing logits to be separated by `input_lens`.
                child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            child_distribution_cls_struct: A struct of Distribution classes that can
                be instantiated from the given logits.
            input_lens: A list or dict of integers that indicate the length of each
                logit. If this is given as a dict, the structure should match the
                structure of child_distribution_cls_struct.
            space: The possibly nested output space.
            **kwargs: Forward compatibility kwargs.

        Returns:
            A TorchMultiActionDistribution object.
        """
        # logit_lens = tree.flatten(input_lens)
        # child_distribution_cls_list = tree.flatten(
        #     child_distribution_cls_struct)
        # split_logits = logits

        # child_distribution_list = tree.map_structure(
        #     lambda dist, input_: dist.from_logits(input_),
        #     child_distribution_cls_list,
        #     list(split_logits),
        # )

        # child_distribution_struct = tree.unflatten_as(
        #     child_distribution_cls_struct, child_distribution_list
        # )

        return CustomDistr(torch.zeros((1, 1)))

    def logp(self, value):
        return torch.rand((1, 1))


class AgentModule(TorchRLModule):
    def __init__(self, config: RLModuleConfig, env_cfg: EnvConfig) -> None:
        super().__init__(config)
        self.agent = Agent(EnvParam.my_player(), env_cfg)
        self._parameters = self.agent.net.state_dict()

    def update_default_view_requirements(self, defaults):
        defaults = super().update_default_view_requirements(defaults)
        defaults['vf_preds'] = ViewRequirement(
            data_col='vf_preds',
            shift=-1,
            used_for_compute_actions=True,
            used_for_training=True,
            batch_repeat_value=1,
        )
        defaults['action_dist_inputs'] = ViewRequirement(
            data_col='action_dist_inputs',
            shift=0,
            used_for_compute_actions=True,
            used_for_training=True,
            batch_repeat_value=1,
            space=gym.spaces.Box(-1, 1, shape=(1,)),
        )
        defaults['action_logp'] = ViewRequirement(
            data_col='action_logp',
            shift=0,
            used_for_compute_actions=True,
            used_for_training=True,
            batch_repeat_value=1,
            space=gym.spaces.Box(-1, 1, shape=(1,)),
        )
        return defaults

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
        logp, logits, value, action, entropy = self.agent.act(batch)
        # sample_action = RayWrapper.get_action_space().sample()
        output = {}
        output['actions'] = action
        output['action_dist_inputs'] = torch.zeros((1, 1))
        output['action_logp'] = logp
        output['vf_preds'] = value
        return output

    def output_specs_exploration(self) -> Mapping[str, gym.spaces.Space]:
        return {
            "actions": NestedDict,
            "vf_preds": torch.Tensor
        }

    # @override(RLModule)
    # def output_specs_train(self) -> dict:
    #     return [
    #         Columns.VF_PREDS,
    #         Columns.ACTION_DIST_INPUTS,
    #     ]
    def get_train_action_dist_cls(self):
        action_space = RayWrapper.get_action_space()
        sample_action = RayWrapper.get_action_space().sample()
        return CustomDistr
    # .get_partial_dist_cls(
    #         child_distribution_cls_struct=tree.map_structure(
    #             lambda x: TorchCategorical, sample_action),
    #         input_lens=tree.map_structure(
    #             lambda x: prod(x.shape), sample_action),
    #         space=action_space
    #     )

    def get_exploration_action_dist_cls(self):
        action_space = RayWrapper.get_action_space()
        sample_action = RayWrapper.get_action_space().sample()
        return CustomDistr
    # .get_partial_dist_cls(
    #         child_distribution_cls_struct=tree.map_structure(
    #             lambda x: TorchCategorical, sample_action),
    #         input_lens=tree.map_structure(lambda x:
    #                                       prod(x.shape), sample_action),
    #         space=action_space
    #     )


class RayWrapper(gym.Env):
    def __init__(self, env_cfg) -> None:
        super().__init__()
        self.lux_env = LuxAI_S2(**env_cfg)
        # fix, because environment doing it wrong
        self.lux_env.env_cfg = EnvConfig.from_dict(
            asdict(self.lux_env.env_cfg))
        self.env_cfg = self.lux_env.env_cfg

        self.my_player = EnvParam.my_player()
        self.opp_player = EnvParam.opp_player()
        self.agent = EarlySetupPlayer(self.my_player, self.env_cfg)
        self.opponent = EarlySetupPlayer(self.opp_player, self.env_cfg)
        self.feature_parser = FeatureParser()
        self.action_parser = ActionParser()
        self._step = 0
        # TODO unused, but throws error if not presetn
        self.observation_space = self.get_observation_space()
        self.action_space = self.get_action_space()
        self.last_raw_obs = None

    @classmethod
    def get_observation_space(cls):
        return get_observation_space()

    @classmethod
    def get_action_space(cls):
        return get_action_space(EnvParam.rule_based_early_step)

    def reset(self):
        _, obs = self._run_bidding_stage()

        self._step = 1  # should be 1 since one step is dedicated to bidding
        while self.lux_env.state.real_env_steps < 0:
            action = {
                self.my_player: self.agent.act(
                    self._step, obs[self.my_player]),
                self.opp_player: self.opponent.act(
                    self._step, obs[self.opp_player])
            }

            self.last_raw_obs, *other = self.lux_env.step(action)
            self._step += 1

        self.last_raw_obs = obs

        (my_obs, opp_obs), global_info = self.feature_parser.parse(
            obs, env_cfg=self.env_cfg)
        # return valid actions from here
        game_state = lux.kit.obs_to_game_state(
            self._step, self.env_cfg, obs[self.my_player])
        valid_actions = get_valid_actions(
            game_state, self.my_player)
        return {'my_obs': my_obs, 'valid_actions': valid_actions}
    # {
    #         "my_obs": my_obs,
    #         "opp_obs": opp_obs,
    #         "valid_actions": valid_actions
    #     }

    def _run_bidding_stage(self):
        obs, _ = self.lux_env.reset()
        my_bid = self.agent.act(
            0, obs[self.my_player])
        opponent_bid = self.opponent.act(
            0, obs[self.opp_player])
        obs, *other = self.lux_env.step({
            self.my_player: my_bid,
            self.opp_player: opponent_bid
        })
        self.last_raw_obs = obs
        game_state = lux.kit.obs_to_game_state(
            0, self.env_cfg, obs[self.my_player])
        return game_state.teams[self.my_player].place_first, obs

    def step(self, raw_action: Any) -> Any:
        action, _ = self.action_parser.parse(
            [
                obs_to_game_state(self._step, self.env_cfg,
                                  self.last_raw_obs[self.my_player]),
            ],
            [raw_action])

        obs, rewards, terminations, truncations, infos = self.lux_env.step(
            action)

        self.last_raw_obs = obs
        self._step += 1
        (my_obs, opp_obs), global_info = self.feature_parser.parse(
            obs, env_cfg=self.env_cfg)
        # return valid actions from here
        game_state = lux.kit.obs_to_game_state(
            self._step, self.env_cfg, obs[self.my_player])
        valid_actions = get_valid_actions(game_state, self.my_player)
        return {'my_obs': my_obs, 'valid_actions': valid_actions}, rewards[self.my_player], terminations[self.my_player] if self._step < 32 else True, infos[self.my_player]

    def render(self, mode: str = 'human') -> Any:
        return self.lux_env.render(mode)


def train():
    # setup config
    config = (PPOConfig()
              .training(sgd_minibatch_size=10)
              .experimental(_enable_new_api_stack=True, _disable_preprocessor_api=True, _disable_initialize_loss_from_dummy_batch=True)
              .environment(env=RayWrapper,
                           disable_env_checking=True,
                           env_config={
                               **asdict(EnvConfig()),
                               "MIN_FACTORIES": EnvParam.MIN_FACTORIES,
                               "MAX_FACTORIES": EnvParam.MAX_FACTORIES,
                               "verbose": 0,
                               "collect_stats": True,
                               "FACTORY_WATER_CONSUMPTION": 0,
                               "max_episode_length": 1000,
                               "BIDDING_SYSTEM": True})
              )

    env_config = config.env_config
    module_spec = SingleAgentRLModuleSpec(
        module_class=partial(
            AgentModule, env_cfg=env_config),
        observation_space=RayWrapper.get_observation_space(),
        action_space=RayWrapper.get_action_space(),
    )
    config = (config
              .rl_module(rl_module_spec=module_spec)
              .training(train_batch_size=10)
              .rollouts(num_rollout_workers=0)
              )
#

    algo = config.build()
    for _ in range(10):
        result = algo.train()
        print(pretty_print(result))


if __name__ == "__main__":
    # install_patch()
    train()
