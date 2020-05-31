import numpy as np
import gym
from env import DmControlEnvForPytorch


class OrdinaryEnvForPytorch(DmControlEnvForPytorch):

    def __init__(self, gym_id, action_repeat=1,
                 obs_type='pixels', render_kwargs=None):
        super(DmControlEnvForPytorch, self).__init__()
        assert obs_type in self.keys
        self.env = gym.make(gym_id)
        self.action_repeat = action_repeat
        self.obs_type = obs_type
        self.render_kwargs = dict(
            width=64,
            height=64,
            camera_id=0,
        )

        if render_kwargs is not None:
            self.render_kwargs.update(render_kwargs)

        if obs_type == 'state':
            self.observation_space = self.env.observation_space
        elif obs_type == 'pixels':
            raise NotImplementedError
        else:
            raise NotImplementedError

        self.action_space = self.env.action_space

    def _preprocess_obs(self, time_step):
        if self.obs_type == 'state':
            obs = time_step[0]
        elif self.obs_type == 'pixels':
            raise NotImplementedError

        return obs

    def _step(self, action):
        time_step = self.env.step(action)
        obs = self._preprocess_obs(time_step)
        reward = time_step[1]
        done = time_step[2]
        message = time_step[3]

        return obs, reward, done, message

    def step(self, action):
        sum_reward = 0.0
        for _ in range(self.action_repeat):
            obs, reward, done, _ = self._step(action)
            sum_reward += reward
            if done:
                break

        return obs, sum_reward, done, None

    def reset(self):
        return self.env.reset()

    def render(self, mode='rgb_array'):
        return self.env.render(mode=mode)

    def seed(self, seed):
        self.env.random = seed




