def get_observation_space():
    obs_space = spaces.Dict(
        {
            'player_0': get_single_observation_space(),
            'player_1': get_single_observation_space()
        }
    )
    return obs_space


def get_single_observation_space():
    global_feature_names = ['env_step',                 # 1000+10
                            'cycle',                    # 20
                            'hour',                     # 50
                            'daytime_or_night',         # 2
                            'num_factory_own',          # 5
                            'num_factory_enm',          # 5
                            'total_lichen_own',         # 48 * 48 * 100
                            'total_lichen_enm',         # 48 * 48 * 100
                            'factory_total_power_own',  # 9999
                            'factory_total_ice_own',    # 9999
                            'factory_total_water_own',  # 9999
                            'factory_total_ore_own',    # 9999
                            'factory_total_metal_own',  # 9999
                            'num_light_own',            # 9999
                            'num_heavy_own',            # 9999
                            'robot_total_power_own',    # 9999
                            'robot_total_ice_own',      # 9999
                            'robot_total_water_own',    # 9999
                            'robot_total_ore_own',      # 9999
                            'robot_total_metal_own',    # 9999
                            'factory_total_power_enm',  # 9999
                            'factory_total_ice_enm',    # 9999
                            'factory_total_water_enm',  # 9999
                            'factory_total_ore_enm',    # 9999
                            'factory_total_metal_enm',  # 9999
                            'num_light_enm',            # 9999
                            'num_heavy_enm',            # 9999
                            'robot_total_power_enm',    # 9999
                            'robot_total_ice_enm',      # 9999
                            'robot_total_water_enm',    # 9999
                            'robot_total_ore_enm',      # 9999
                            'robot_total_metal_enm']    # 9999
    global_feature_space = [
        1000,
        20,
        50,
        2,
        5,
        5,
        230400,
        230400,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999,
        9999
    ]
    global_feature_space = spaces.MultiDiscrete(
        np.array(global_feature_space), dtype=np.float64)

    map_featrue_names = {
        'ice': 9999,
        'ore': 9999,
        'rubble': 100,
        'lichen': 100,
        'lichen_strains': 11,  # -1, 0, 1
        'lichen_strains_own': 2,
        'lichen_strains_enm': 2,
        'valid_region_indicator': 2,
        'factory_id': 10,
        'factory_power': 9999,
        'factory_ice': 9999,
        'factory_water': 9999,
        'factory_ore': 9999,
        'factory_metal': 9999,
        'factory_own': 2,
        'factory_enm': 2,
        'factory_can_build_light': 2,
        'factory_can_build_heavy': 2,
        'factory_can_grow_lichen': 2,
        'factory_water_cost': 9999,
        'unit_id': 9999,
        'unit_power': 9999,
        'unit_ice': 9999,
        'unit_water': 9999,
        'unit_ore': 9999,
        'unit_metal': 9999,
        'unit_own': 2,
        'unit_enm': 2,
        'unit_light': 2,
        'unit_heavy': 2
    }
    map_featrue_space = np.tile(
        np.array(list(map_featrue_names.values())).reshape(30, 1, 1), (1, 48, 48))
    map_featrue_space = spaces.MultiDiscrete(
        map_featrue_space, dtype=np.float64)

    action_feature_space = spaces.Dict(
        {
            'unit_indicator': spaces.MultiDiscrete(np.full((48, 48), 2), dtype=np.float64),
            'type': spaces.MultiDiscrete(np.full((48, 48, 20), 7), dtype=np.float64),
            'direction': spaces.MultiDiscrete(np.full((48, 48, 20), 5), dtype=np.float64),
            'resource': spaces.MultiDiscrete(np.full((48, 48, 20), 5), dtype=np.float64),
            'amount': spaces.MultiDiscrete(np.full((48, 48, 20), 10), dtype=np.float64),
            'repeat': spaces.MultiDiscrete(np.full((48, 48, 20), 2), dtype=np.float64),
            'n': spaces.MultiDiscrete(np.full((48, 48, 20), 9999), dtype=np.float64)
        }
    )
    obs_space = spaces.Dict(
        {
            'global_feature': global_feature_space,
            'map_feature': map_featrue_space,
            'action_feature': action_feature_space
        }
    )
    return obs_space
