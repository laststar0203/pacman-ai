import gym

import random
import collections

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

HyperParameter = collections.namedtuple('HyperParameter', ['batch_size', 'gamma', 'learning_rate', 'buffer_limit'])


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

        self._ln1 = nn.Linear(2592, 9)

    def forward(self, x):
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))

        x = x.view(-1) if x.dim() == 3 else x.view(x.shape[0], -1)

        x = self._ln1(x)

        return x


class ReplayBuffer:

    def __init__(self, parameter: HyperParameter):
        self._buffer = collections.deque(maxlen=parameter.buffer_limit)

    @property
    def size(self):
        return len(self._buffer)

    def put(self, state, state_prime, action, reward, done):
        self._buffer.append((state, state_prime, action, reward, done))

    def sample(self, n):
        mini_batch = random.sample(self._buffer, n)
        state_list, action_list, reward_list, state_prime_list, done_mask_list = [], [], [], [], []

        for transition in mini_batch:
            state, state_prime, action, reward, done_mask = transition

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

        self._memory = ReplayBuffer(parameter.buffer_limit)

        self._optimizer = optim.Adam(self._q_network.parameters(), lr=parameter.learning_rate)

        self.update_network()

    @property
    def memory(self):
        return self._memory

    def update_network(self):
        self._target_network.load_state_dict(self._q_network.state_dict())

    def predict(self, state, epsilon):

        out = self._q_network(state)
        r = random.random()

        # epsilon greedy
        if r < epsilon:
            return random.randint(0, 8)
        else:
            return out.argmax().item()

    def train(self):

        state_list, action_list, reward_list, state_prime_list, \
        done_mask_list = self._memory.sample(self._PARAMETER.batch_size)

        output = self._q_network(state_list)
        q_action = output.gather(1, action_list)

        max_q_prime = self._target_network(state_prime_list).max(1)[0].unsqueeze(1)
        target = reward_list + self._PARAMETER.gamma * max_q_prime * done_mask_list

        loss = F.smooth_l1_loss(q_action, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def save(self):
        pass


class Agent:

    def __init__(self, environment, algorithm: DQN):
        self._environment = environment
        self._algorithm = algorithm

    @property
    def algorithm(self):
        return self._algorithm

    def step(self, state, epsilon):
        action = self._algorithm.predict(state=state, epsilon=epsilon)
        state_prime, reward, done, info = self._environment.step(action, self)

        self._algorithm.memory.put(
            state=state,
            state_prime=state_prime,
            action=action,
            reward=reward,
            done=done
        )

        if self._algorithm.memory.size > 2000:
            self._algorithm.train()

        return state_prime, reward, done, info

    def save(self):
        self._algorithm.save()


class Environment(gym.Wrapper):
    move = -1
    eat = 50
    death = -1000

    def __init__(self):
        super(Environment, self).__init__(gym.make('MsPacman-v0'))

        self._move_reward = Environment.move
        self._eat_reward = Environment.eat
        self._death_reward = Environment.death

        self._metadata = None

    def reset(self,
              reward_move: int = move,
              reward_eat: int = eat,
              reward_death: int = death,
              **kwargs):

        self._move_reward = reward_move
        self._eat_reward = reward_eat
        self._death_reward = reward_death

        state = super(Environment, self).reset(**kwargs)
        return self.observation(state)

    def step(self, action):
        state_prime, reward, done, info = super(Environment, self).step(action)

        state_prime = self.observation(state_prime)
        reward = self.reward(reward, info)

        self._metadata = info

        return state_prime, reward, done, info

    def reward(self, reward, info):

        new_reward = 0

        # move
        if reward == 0:
            new_reward = self._move_reward
        # eat
        elif reward == 10:
            new_reward = self._eat_reward

        if self._metadata and self._metadata['lives'] > info['lives']:
            new_reward -= 1000

        return new_reward

    def observation(self, observation):
        observation = observation[1:172, 1:160]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])

        return transform(observation)


if __name__ == '__main__':

    parameter = HyperParameter(
        batch_size=32,
        buffer_limit=50000,
        gamma=0.98,
        learning_rate=0.1
    )

    env = Environment()
    dqn = DQN(parameter=parameter)

    agent = Agent(environment=env, algorithm=dqn)

    print_interval = 20
    score = 0.0

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))

        state = env.reset()
        done = False

        while not done:
            state_prime, reward, done, info = agent.step(state, epsilon)
            state = state_prime

            score += reward

        if n_epi % print_interval == 0 and n_epi != 0:
            dqn.update_network()
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size, epsilon * 100))
            score = 0.0
