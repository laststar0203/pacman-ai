import gym


class Environment(gym.Wrapper):
    _move = -1
    _eat = 50
    _death = -1000

    def __init__(self, env):
        super(Environment, self).__init__(env)

        self._move_reward = Environment._move
        self._eat_reward = Environment._eat
        self._death_reward = Environment._death



    def reset(self,
              reward_move: int = _move,
              reward_eat: int = _eat,
              reward_death: int = _death,
              **kwargs):

        self._move_reward = reward_move
        self._eat_reward = reward_eat
        self._death_reward = reward_death

        super(Environment, self).reset(**kwargs)

    def step(self, action, agent=None):
        state_prime, reward, done, info = super(Environment, self).step(action)
        return state_prime, self.reward(reward, info), done

    def reward(self,
               reward: int,
               info: dict):

        if reward == 0:
            pass

        # move
        if reward == 0:
            pass

        # eat
        if reward == 10:
            pass

        # death
        if reward

    def observation(self, observation):
        pass


class Agent(gym.ActionWrapper):

    def __init__(self, env):
        super().__init__(env)

        self._heart = None


    def action(self, action):
        pass

    def reverse_action(self, action):
        pass
