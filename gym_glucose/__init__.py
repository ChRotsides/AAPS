from gym.envs.registration import register
register(id='MyGym-v0', 
    entry_point='gym_glucose.envs:MyGym', 
)
