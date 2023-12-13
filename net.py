# from typing import Tuple
# import gym
# import torch
# import torch as th
# import torch.nn as nn
# import torch.nn.functional as F

# from stable_baselines3.common.policies import ActorCriticPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3 import A2C
# from dataclasses import dataclass


# class ActDim:

#     def __init__(self, map_size) -> None:
#         self.BID = 10
#         self.SPAWN_LOCATION = map_size
#         self.SPAWN_WATER = 7
#         self.SPAWN_METAL = 7
#         self.FACTORY = 48 * 7


# # Because of that it can decide e.g. robot A goes north, robot B goes south with probabilities close to one. I
# #  do it this way to keep the action space simple (and the heavy lifting of coordinating units is done by the model).

# class MimicPolicy(ActorCriticPolicy):
#     def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
#         super().__init__(sess, ob_space, ac_space, n_env,
#                          n_steps, n_batch, reuse=reuse, scale=True)
#         self.features_dim = 64
#         self.critic = self.make_critic()

#         self.bid_act = nn.Linear(self.features_dim, 10)

#     def make_critic(self):

#         return nn.Sequential(
#             nn.Conv2d(1, 1, (3, 3), 2),
#             nn.GELU(),
#             nn.Conv2d(1, 1, (3, 3), 2),
#             nn.GELU(),
#             nn.AdaptiveAvgPool3d(self.features_dim),
#             nn.Linear(self.features_dim, self.features_dim),
#             nn.Linear(self.features_dim, 1)
#         )

#     def forward(self, obs: nn.Tensor, deterministic: bool = False) -> Tuple[nn.Tensor, nn.Tensor, nn.Tensor]:
#         features = self.extract_features(obs)
#         value = self.critic(features)
#         return super().forward(obs, deterministic)

#     #     with tf.variable_scope("model", reuse=reuse):
#     #         activ = tf.nn.relu

#     #         extracted_features = nature_cnn(self.processed_obs, **kwargs)
#     #         extracted_features = tf.layers.flatten(extracted_features)

#     #         pi_h = extracted_features
#     #         for i, layer_size in enumerate([128, 128, 128]):
#     #             pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
#     #         pi_latent = pi_h

#     #         vf_h = extracted_features
#     #         for i, layer_size in enumerate([32, 32]):
#     #             vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
#     #         value_fn = tf.layers.dense(vf_h, 1, name='vf')
#     #         vf_latent = vf_h

#     #         self._proba_distribution, self._policy, self.q_value = \
#     #             self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

#     #     self._value_fn = value_fn
#     #     self._setup_init()

#     # def step(self, obs, state=None, mask=None, deterministic=False):
#     #     if deterministic:
#     #         action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
#     #                                                {self.obs_ph: obs})
#     #     else:
#     #         action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
#     #                                                {self.obs_ph: obs})
#     #     return action, value, self.initial_state, neglogp

#     # def proba_step(self, obs, state=None, mask=None):
#     #     return self.sess.run(self.policy_proba, {self.obs_ph: obs})

#     # def value(self, obs, state=None, mask=None):
#     #     return self.sess.run(self.value_flat, {self.obs_ph: obs})


# # # Create and wrap the environment
# # env = DummyVecEnv([lambda: gym.make('Breakout-v0')])

# # model = A2C(CustomPolicy, env, verbose=1)
# # # Train the agent
# # model.learn(total_timesteps=100000)
