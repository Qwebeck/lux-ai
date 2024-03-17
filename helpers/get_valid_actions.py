import lux.kit
from impl_config import ActDims, EnvParam, FactoryActType, UnitActType
from lux.utils import my_turn_to_place_factory


import numpy as np
import torch
import tree
from luxai_s2.actions import move_deltas


import dataclasses


factory_adjacent_delta_xy = np.array([
    [-2, -1],
    [-2, +0],
    [-2, +1],
])
factory_adjacent_delta_xy = np.concatenate(
    [factory_adjacent_delta_xy, -factory_adjacent_delta_xy])
factory_adjacent_delta_xy = np.concatenate(
    [factory_adjacent_delta_xy, factory_adjacent_delta_xy[:, ::-1]])


def get_valid_actions(game_state: lux.kit.GameState, player_id: int):
    player = 'player_0' if player_id == 0 else 'player_1'
    enemy = 'player_1' if player_id == 0 else 'player_0'
    board = game_state.board
    env_cfg = game_state.env_cfg

    def factory_under_unit(unit_pos, factories):
        for _, factory in factories.items():
            factory_pos = factory.pos
            if abs(unit_pos[0] - factory_pos[0]) <= 1 and abs(unit_pos[1] - factory_pos[1]) <= 1:
                return factory
        return None

    act_dims_mapping = dataclasses.asdict(EnvParam.act_dims_mapping)

    valid_actions = tree.map_structure(
        lambda dim: np.zeros((dim, EnvParam.map_size, EnvParam.map_size), dtype=np.bool8), act_dims_mapping)

    factory_va = valid_actions["factory_act"]
    for unit_id, factory in game_state.factories[player].items():
        x, y = factory.pos
        # valid build light
        if factory.cargo.metal >= env_cfg.ROBOTS['LIGHT'].METAL_COST\
                and factory.power >= env_cfg.ROBOTS['LIGHT'].POWER_COST:
            factory_va[FactoryActType.BUILD_LIGHT, x, y] = True
        # valid build heavy
        if factory.cargo.metal >= env_cfg.ROBOTS['HEAVY'].METAL_COST\
                and factory.power >= env_cfg.ROBOTS['HEAVY'].POWER_COST:
            factory_va[FactoryActType.BUILD_HEAVY, x, y] = True
        # valid grow lichen
        lichen_strains_size = np.sum(board.lichen_strains == factory.strain_id)
        if factory.cargo.water >= (lichen_strains_size + 1) // env_cfg.LICHEN_WATERING_COST_FACTOR + 1:
            adj_xy = factory.pos + factory_adjacent_delta_xy
            adj_xy = adj_xy[(adj_xy >= 0).all(axis=1) & (
                adj_xy < EnvParam.map_size).all(axis=1)]
            adj_x, adj_y = adj_xy[:, 0], adj_xy[:, 1]
            no_ruble = (board.rubble[adj_x, adj_y] == 0)
            no_ice = (board.ice[adj_x, adj_y] == 0)
            no_ore = (board.ore[adj_x, adj_y] == 0)
            if (no_ruble & no_ice & no_ore).any():
                factory_va[FactoryActType.WATER, x, y] = True

        # always can do nothing
        factory_va[FactoryActType.DO_NOTHING, x, y] = True

    # construct unit_map
    unit_map = np.full_like(game_state.board.rubble,
                            fill_value=-1, dtype=np.int32)
    for unit_id, unit in game_state.units[player].items():
        x, y = unit.pos
        unit_map[x, y] = int(unit_id[len("unit_"):])

    for unit_id, unit in game_state.units[player].items():
        x, y = unit.pos
        action_queue_cost = unit.action_queue_cost(game_state)
        if unit.power >= action_queue_cost:
            valid_actions["unit_act"]["act_type"][:, x, y] = True
        else:
            valid_actions["unit_act"]["act_type"][UnitActType.DO_NOTHING, x, y] = True
            continue

        # valid unit move
        valid_actions["unit_act"]["move"]["repeat"][:, x, y] = True
        for direction in range(len(move_deltas)):
            target_pos = unit.pos + move_deltas[direction]

            # always forbid to move to the same position
            if direction == 0:
                continue

            if (target_pos[0] < 0 or target_pos[1] < 0 or target_pos[0] >= EnvParam.map_size
                    or target_pos[1] >= EnvParam.map_size):
                continue

            if factory_under_unit(target_pos, game_state.factories[enemy]) is not None:
                continue

            power_required = unit.move_cost(game_state, direction)
            if unit.power - action_queue_cost >= power_required:
                valid_actions["unit_act"]["move"]["direction"][direction, x, y] = True

        # valid transfer
        valid_actions["unit_act"]["transfer"]['repeat'][0, x, y] = True
        amounts = [unit.cargo.ice, unit.cargo.ore,
                   unit.cargo.water, unit.cargo.metal, unit.power]
        for i, a in enumerate(amounts):
            valid_actions["unit_act"]["transfer"]['resource'][i, x, y] = (
                a > 0)
        for direction in range(1, len(move_deltas)):
            target_pos = unit.pos + move_deltas[direction]

            # always forbid to transfer to self
            if direction == 0:
                continue

            there_is_a_target = False
            if (target_pos >= 0).all() and (target_pos < env_cfg.map_size).all():
                there_is_a_target = (
                    unit_map[target_pos[0], target_pos[1]] != -1)
            if factory_under_unit(target_pos, game_state.factories[player]) is not None:
                there_is_a_target = True

            if there_is_a_target:
                valid_actions["unit_act"]["transfer"]["direction"][direction, x, y] = True

        # valid pickup
        valid_actions["unit_act"]["pickup"]['repeat'][0, x, y] = True
        factory = factory_under_unit(unit.pos, game_state.factories[player])
        if factory is not None:
            valid_actions["unit_act"]["act_type"][UnitActType.PICKUP, x, y] = True
            amounts = [
                factory.cargo.ice, factory.cargo.ore, factory.cargo.water, factory.cargo.metal, factory.power
            ]
            for i, a in enumerate(amounts):
                valid_actions["unit_act"]["pickup"]['resource'][i, x, y] = (
                    a > 0)

        # valid dig
        if factory_under_unit(unit.pos, game_state.factories[player]) is None \
                and unit.power - action_queue_cost >= unit.unit_cfg.DIG_COST:
            if (board.lichen[x, y] > 0) or (board.rubble[x, y] > 0) \
                    or (board.ice[x, y] > 0) or (board.ore[x, y] > 0):
                valid_actions["unit_act"]["dig"]['repeat'][:, x, y] = True

        # valid selfdestruct
        if unit.power - action_queue_cost >= unit.unit_cfg.SELF_DESTRUCT_COST:
            # self destruct can not repeat
            valid_actions["unit_act"]["self_destruct"]['repeat'][0, x, y] = True

        # valid recharge
        valid_actions["unit_act"]["recharge"]['repeat'][0, x, y] = True

    # calculate va for the flattened action space
    move_va = valid_actions["unit_act"]["move"]
    move_va = valid_actions["unit_act"]["act_type"][UnitActType.MOVE][None, None] \
        & move_va['direction'][:, None] \
        & move_va['repeat'][None, :]  # 5*2=10

    transfer_va = valid_actions["unit_act"]["transfer"]
    transfer_va = valid_actions["unit_act"]["act_type"][UnitActType.TRANSFER][None, None, None] \
        & transfer_va['direction'][:, None, None] \
        & transfer_va['resource'][None, :, None] \
        & transfer_va['repeat'][None, None, :]  # 5*5*2=50

    pickup_va = valid_actions["unit_act"]["pickup"]
    pickup_va = valid_actions["unit_act"]["act_type"][UnitActType.PICKUP][None, None] \
        & pickup_va['resource'][:, None] \
        & pickup_va['repeat'][None, :]  # 5*2=10

    dig_va = valid_actions["unit_act"]["act_type"][UnitActType.DIG][None] \
        & valid_actions["unit_act"]["dig"]['repeat']  # 2

    self_destruct_va = valid_actions["unit_act"]["act_type"][UnitActType.SELF_DESTRUCT][None] \
        & valid_actions["unit_act"]["self_destruct"]['repeat']  # 2

    recharge_va = valid_actions["unit_act"]["act_type"][UnitActType.RECHARGE][None] \
        & valid_actions["unit_act"]["recharge"]['repeat']  # 2

    # 1
    do_nothing_va = valid_actions["unit_act"]["act_type"][UnitActType.DO_NOTHING]

    valid_actions = {}
    if not EnvParam.rule_based_early_step:
        if game_state.env_steps == 0:
            bid_va = np.ones(ActDims.bid, dtype=np.bool8)
        else:
            bid_va = np.zeros(ActDims.bid, dtype=np.bool8)

        if game_state.env_steps != 0 \
                and game_state.real_env_steps < 0 \
                and my_turn_to_place_factory(game_state.teams[player].place_first, game_state.env_steps):
            factory_spawn = board.valid_spawns_mask
        else:
            factory_spawn = np.zeros_like(
                board.valid_spawns_mask, dtype=np.bool8)

        valid_actions = {
            "bid": bid_va,
            "factory_spawn": factory_spawn,
        }

    valid_actions.update({
        "factory_act": factory_va,
        "move": move_va,
        "transfer": transfer_va,
        "pickup": pickup_va,
        "dig": dig_va,
        "self_destruct": self_destruct_va,
        "recharge": recharge_va,
        "do_nothing": do_nothing_va,
    })
    return valid_actions
