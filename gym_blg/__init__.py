from gym.envs.registration import register

register(
    id='blgd-v0',
    entry_point='gym_blg.envs:BlgDiscreteEnv',
)