import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQN:
    pass

class ReplayBuffer:
    pass


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()
        self._conv1 = nn.Conv2d(in_channels=3,
                                out_channels=16,
                                kernel_size=8,
                                stride=4)
        self._conv2 = nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=4,
                                stride=2)
        self._ln1 = nn.Linear()




class Agent:
    def __init__(self):
        self.life = 3
        self.is_death = False


class Environment(gym.Wrapper):
    _move = -1
    _eat = 50
    _death = -1000

    def __init__(self):

        env = gym.make('MsPacman-v0')

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

        self._update_agent(info, agent)

        return state_prime, self.reward(reward, agent), done

    def _update_agent(self, info, agent):

        if info['life'] < agent.life:
            agent.is_death = True
            agent.life = info['life']
        else:
            agent.is_death = False

    def reward(self,
               reward: int,
               agent: Agent):

        current_reward = 0

        # move
        if reward == 0:
            current_reward = -1
        elif reward == 10:
            current_reward = 50

        if agent.is_death:
            current_reward -= 1000


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    for n_epi in range(1000):
        pass
