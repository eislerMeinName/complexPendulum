import gymnasium as gym

gym.envs.register(
        id='complexPendulum-v0',
        entry_point='complexPendulum.envs:ComplexPendulum',
    )
