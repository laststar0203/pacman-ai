import gym

import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

HyperParameter = collections.namedtuple('HyperParameter', ['batch_size', 'gamma', 'learning_rate'])


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

    def forward(self):
        pass


class ReplayBuffer:

    def __init__(self, buffer_limit):
        self._buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self._buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self._buffer, n)
        state_list, action_list, reward_list, state_prime_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            state, action, reward, state_prime, done_mask = transition

            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            state_prime_list.append(state_prime)
            done_mask_list.append([done_mask])

        return torch.stack(state_list), torch.tensor(action_list), \
               torch.tensor(reward_list), torch.stack(state_prime_list), \
               torch.tensor(done_mask_list)


class DQN:

    def __init__(self, parameter: HyperParameter):
        self._PARAMETER = parameter

        self._q_network = Qnet()
        self._target_network = Qnet()
        self._replay_buffer = ReplayBuffer(buffer_limit=parameter.batch_size)

        self._optimizer = optim.Adam(self._q_network.parameters(), lr=parameter.learning_rate)

    def behavior(self, state, epsilon):

        out = self._q_network(state)
        r = random.random()

        # epsilon greedy
        if r < epsilon:
            return random.randint(0, 8)
        else:
            return out.argmax().item()

    def memorize(self, state, action, reward, state_prime, done):

        # state 전처리
        self._replay_buffer.put((state, action, reward, state_prime, done))

    def train(self):
        state_list, action_list, reward_list, state_prime_list, \
        done_mask_list = self._replay_buffer.sample(self._PARAMETER.batch_size)

        output = self._q_network(state_list)
        q_action = output.gather(1, action_list)

        max_q_prime = self._target_network(state_prime_list).max(1)[0].unsqueeze(1)
        target = reward_list + self._PARAMETER.gamma * max_q_prime * done_mask_list

        loss = F.smooth_l1_loss(q_action, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


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
            current_reward = Environment._move
        # eat
        elif reward == 10:
            current_reward = Environment._eat

        if agent.is_death:
            current_reward = Environment._death

        return current_reward

if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    for n_epi in range(1000):
        pass
