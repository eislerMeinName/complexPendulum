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

gym.envs.register(
        id='baselinePendulum-v0',
        entry_point='complexPendulum.envs:BaselineGainPendulum'
    )

gym.envs.register(
        id='easybaselinePendulum-v0',
        entry_point='complexPendulum.envs:EasyBaselinePendulum'
    )

gym.envs.register(
        id='directUPPendulum-v0', 
        entry_point='complexPendulum.envs:DirectUpPendulum'
    )
