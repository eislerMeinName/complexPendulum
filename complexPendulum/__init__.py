import gymnasium as gym

gym.envs.register(
        id='complexPendulum-v0',
        entry_point='complexPendulum.envs:ComplexPendulum',
    )

gym.envs.register(
        id='directPendulum-v0',
        entry_point='complexPendulum.envs:DirectPendulum',
    )

gym.envs.register(
        id='gainPendulum-v0',
        entry_point='complexPendulum.envs:GainPendulum',
    )
