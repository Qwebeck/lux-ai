from lux.kit import obs_to_game_state, GameState, EnvConfig
from lux.utils import my_turn_to_place_factory
import numpy as np
from scipy import ndimage
import scipy


class Player:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        k = 200
        ice_log_weight = 1
        ore_log_weight = 0.5
        rubble_weight = 0.1
        sigma = 3  # ??
        if step == 0:
            # random bid
            bid_choice = [-2, 0, 2]
            bid_num = np.random.choice(bid_choice)
            return dict(faction="AlphaStrike", bid=bid_num)
        else:
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            H, W = obs['board']['ice'].shape
            real_env_steps = game_state.real_env_steps
            water_left = game_state.teams[self.player].water
            metal_left = game_state.teams[self.player].metal

            factories_to_place = game_state.teams[self.player].factories_to_place
            is_my_turn_to_place_factory = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step)

            if factories_to_place > 0 and is_my_turn_to_place_factory:
                valid_spawns_mask = game_state.board.valid_spawns_mask

                rubble_0 = (game_state.board.rubble == 0).astype(np.int32)
                # Check math
                center_weight = ndimage.gaussian_filter(np.array([[1.]], dtype=np.float32),
                                                        sigma=sigma, mode='constant', cval=0.0)
                ice_sum = ndimage.gaussian_filter(game_state.board.ice.astype(
                    np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                ore_sum = ndimage.gaussian_filter(game_state.board.ore.astype(
                    np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                rubble_sum = ndimage.gaussian_filter(rubble_0.astype(
                    np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight
                factory_occupancy_map = game_state.board.factory_occupancy_map
                factory_occupancy_map = factory_occupancy_map == int(
                    self.player[len('player_'):])
                factory_occupancy_map = ndimage.gaussian_filter(factory_occupancy_map.astype(
                    np.float32), sigma=sigma, mode='constant', cval=0.0) / center_weight

                ice_sum = np.minimum(ice_sum, 3)
                ore_sum = np.minimum(ore_sum, 3)
                factory_occupancy_map = np.minimum(factory_occupancy_map, 3)

                score = sum([
                    np.log(ice_sum + 0.2) * ice_log_weight,
                    np.log(ore_sum + 0.2) * ore_log_weight,
                    np.log(rubble_sum + 1) * rubble_weight,
                    -np.log(factory_occupancy_map + 0.2),
                    np.log(valid_spawns_mask + np.finfo(np.float64).tiny),
                ])

                topk_idx = np.argsort(score.flat)[-k:]
                topk_score = score.flat[topk_idx]

                # sample one from top-k candidates
                # pi = topk_score/topk_score.sum()
                pi = scipy.special.softmax(topk_score)
                idx = np.random.choice(topk_idx, p=pi)
                spawn_loc = [idx // W, idx % W]

                # post processing: shift to local maxima
                while True:
                    i, j = spawn_loc
                    cur_score = score[i, j]
                    max_score = score[i, j]
                    for di, dj in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
                        if 0 <= i + di < H and 0 <= j + dj < W and score[i + di, j + dj] > max_score:
                            max_score = score[i + di, j + dj]
                            spawn_loc = [i + di, j + dj]
                    if max_score == cur_score:
                        break

                if game_state.teams[self.player].place_first and real_env_steps == -2:
                    return dict(spawn=spawn_loc, metal=metal_left, water=water_left)
                elif ~game_state.teams[self.player].place_first and real_env_steps == -1:
                    return dict(spawn=spawn_loc, metal=metal_left, water=water_left)
                else:
                    metal_num = np.random.randint(100, 200)
                    water_num = np.random.randint(100, 200)
                    return dict(spawn=spawn_loc, metal=metal_num, water=water_num)
        return dict()
