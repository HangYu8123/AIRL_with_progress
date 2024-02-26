# import gym

# gym.logger.set_level(40)


# def make_env(env_id):
#     return NormalizedEnv(gym.make(env_id))


# class NormalizedEnv(gym.Wrapper):

#     def __init__(self, env):
#         gym.Wrapper.__init__(self, env)
#         self._max_episode_steps = env._max_episode_steps

#         self.scale = env.action_space.high
#         self.action_space.high /= self.scale
#         self.action_space.low /= self.scale

#     def step(self, action):
#         return self.env.step(action * self.scale)
class kortex_arm:
    def __init__(self, observation_space, action_space, cup_id=2, seed=0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed
        np.random.seed(seed)
        init_state(cup_id)
    def init_state(self, cup_id):
        if cup_id == 0:
            self.init_state = ?
        elif cup_id == 1:
            self.init_state = ?
        elif cup_id == 2:
            self.init_state = ?
        else:
            self.init_state = ?

    def seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
    
    def reset(self):
        return self.init_state
    def step(self, action):
        # 判断是否到了target地点
        return state, reward, done, _