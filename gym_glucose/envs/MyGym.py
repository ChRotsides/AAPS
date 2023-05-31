import gym
from gym_glucose.envs.GlucoseEnv import GlucoseEnvironment
from gym import spaces
import numpy as np

class MyGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        self.env = GlucoseEnvironment()
        self.action_space = spaces.Discrete(100)  # Modify this as per your requirement
        self.observation_space = spaces.Dict({
            'obs1': spaces.Box(low=np.zeros((20,3)), high=np.ones((20,3))),
            'obs2': spaces.Box(low=np.zeros((1,20)), high=np.ones((1,20))),
        })
        self.current_step = 0

    def step(self, action):
        ob, extra_info,reward, done = self._take_action(action)
        # reward = self._get_reward(ob, action)
        self.current_step += 1

        if self.current_step > 100:  # Suppose we want to stop after 100 steps
            done = True
        return ob, extra_info, reward, done
        # return [ob, extra_info], reward, done, {}

    def reset(self):
        self.current_step = 0
        ob,extra_info,reward,done=self.env.reset()
        return ob,extra_info

    def render(self, mode='human', close=False):
        # Implement your render logic here
        pass

    def _take_action(self, action):
        # print(action)
        return self.env.step(action*0.002)

    def _get_reward(self, ob, action):
        # Implement your reward logic here
        # Suppose, for instance, the reward is directly the glucose level
        glucose_level = ob[0]  # Or some other way of accessing the glucose level
        return glucose_level
