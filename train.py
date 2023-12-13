from typing import Any
from luxai_s2.state.state import ObservationStateDict
from luxai_s2.spaces.obs_space import get_obs_space
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import BaseFeaturesExtractor
import gym
import numpy as np

from lux.utils import my_turn_to_place_factory
from lux.unit import Unit

"""
Stages: 
1. model responsible for control of one single robot 
2. implement some method for communication
"""


# dict(vf=[<value layer sizes>], pi=[<policy layer sizes>])


class SingleRobotController(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.env = env
        self.prev_step_metrics = None

        directions = ['left', 'top', 'right', 'bottom', 'self']
        resources = ['water', 'metal', 'ore', 'ice', 'power']

        self.action_space = gym.spaces.Discrete(
            len(directions) * len(resources)  # transfer
            +
            len(directions)  # move
            +
            1  # recharge
        )

        self.observation_space = gym.spaces.Box(0, 100, shape=(10, 10))
        # get_obs_space(
        #     config=self.env_cfg, agent_names=self.possible_agents
        # )

    def factory_placement_policy(player, obs: ObservationStateDict):
        potential_spawns = np.array(
            list(
                zip(*np.where(obs["board"]["valid_spawns_mask"] == 1)))
        )
        spawn_loc = potential_spawns[
            np.random.randint(0, len(potential_spawns))
        ]
        return dict(spawn=spawn_loc, metal=150, water=150)

    def bid_policy(player, obs: ObservationStateDict):
        faction = "AlphaStrike"
        if player == "player_1":
            faction = "MotherMars"
        return dict(bid=0, faction=faction)

    def action_to_lux_action(self, action):

        return np.array()

    def step(self, action: int):
        agent = "player_0"
        lux_action = dict()
        for agent in self.env.agents:
            lux_action[agent] = self.action_to_lux_action(action)
        obs, reward, done, info = self.env.step(lux_action)
        return obs, reward, done, info

    # not controlling factory placement

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        action = {}
        for agent in self.env.agents:
            action[agent] = self.bid_policy(agent, obs)
        obs, _, _, _ = self.env.step(action)
        while self.env.state.real_env_steps < 0:
            action = {}
            for agent in self.env.agents:
                if my_turn_to_place_factory(obs["player_0"]["teams"][agent]["place_first"], self.env.state.env_steps):
                    action[agent] = self.factory_placement_policy(
                        agent, obs[agent])
                else:
                    action[agent] = {}
        obs, _, _, _ = self.env.step(action)
        return obs


def make_env(max_episode_steps=1000):
    env_id = "LuxAI_S2-v0"
    env = gym.make(env_id, verbose=0, collect_stats=True,
                   FACTORY_WATER_CONSUMPTION=0, MAX_FACTORIES=3)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = Monitor(env)
    env = SingleRobotController(env)
    return env


def train():
    # Parallel environments
    env = make_env()
    model = A2C("MlpPolicy", env, verbose=1)
    print(model.policy)
    # model.learn(total_timesteps=25000)
    # model.save("a2c_mimic")


if __name__ == "__main__":
    gym.register(
        id="LuxAI_S2-v0",
        entry_point="luxai_s2.env:LuxAI_S2",
    )
    train()
