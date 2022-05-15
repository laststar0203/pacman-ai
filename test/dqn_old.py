import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

# Hyperparameters
learning_rate = 0.1
gamma = 0.98
buffer_limit = 50000
batch_size = 32


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
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

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self._conv1 = nn.Conv2d(3, 16, (5, 3))
        # self.bn1 = nn.BatchNorm2d(16)
        self._max_pool1 = nn.MaxPool2d(kernel_size=(3, 2))

        self._conv2 = nn.Conv2d(16, 32, (5, 3))
        self._max_pool2 = nn.MaxPool2d(kernel_size=(3, 2))
        # self.bn2 = nn.BatchNorm2d(32)
        self._conv3 = nn.Conv2d(32, 32, (5, 3))
        # self.bn3 = nn.BatchNorm2d(32)
        self._max_pool3 = nn.MaxPool2d(kernel_size=(3, 2))

        self._ln1 = nn.Linear(2304, 512)
        self._ln2 = nn.Linear(512, 256)
        self._ln3 = nn.Linear(256, 64)
        self._ln4 = nn.Linear(64, 9)

    def forward(self, x):

        x = self._conv1(x)
        x = F.relu(x)

        x = self._max_pool1(x)

        x = self._conv2(x)
        x = F.relu(x)

        x = self._max_pool2(x)

        x = self._conv3(x)
        x = F.relu(x)

        x = self._max_pool3(x)

        if x.dim() == 3:
            x = x.view(-1)
        else:
            x = x.view(batch_size, -1)

        x = self._ln1(x)
        x = F.relu(x)

        x = self._ln2(x)
        x = F.relu(x)

        x = self._ln3(x)
        x = F.relu(x)

        x = self._ln4(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, 8)
        else:
            return out.argmax().item()


def train(q_network, target_network, memory, optimizer):
    state_list, action_list, reward_list, state_prime_list, \
    done_mask_list = memory.sample(batch_size)

    output = q_network(state_list)
    q_action = output.gather(1, action_list)


    max_q_prime = target_network(state_prime_list).max(1)[0].unsqueeze(1)
    target = reward_list + gamma * max_q_prime * done_mask_list
    loss = F.smooth_l1_loss(q_action, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def preproess_state(state):
    state = state[1:172, 1:160]
    r = state[:, :, 0]
    g = state[:, :, 1]
    b = state[:, :, 2]

    return np.asarray([r, g, b])


if __name__ == '__main__':
    env = gym.make('MsPacman-v0')

    q_network = Qnet()
    target_network = Qnet()
    target_network.load_state_dict(q_network.state_dict())

    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0
    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)

    render = False

    for n_epi in range(10000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))

        state = env.reset()
        state = torch.tensor(preproess_state(state)).float()
        done = False

        while not done:

            # 순전파를 통한 액션 도출
            action = q_network.sample_action(state, epsilon)
            state_prime, reward, done, info = env.step(action)

            reward = -1 if reward == 0 else reward

            state_prime = torch.tensor(preproess_state(state_prime)).float()

            done_mask = 0.0 if done else 1.0

            memory.put((state, action, reward, state_prime, done_mask))
            state = state_prime

            score += reward

            if render:
                env.render()
                import time
                time.sleep(0.01)

            if done:
                break

        if memory.size() > 2000:
            # 순전파, 역전파를 통한 학습
            train(q_network, target_network, memory, optimizer)

        if score / print_interval > 700:
            render = True


        if n_epi % print_interval == 0 and n_epi != 0:
            target_network.load_state_dict(q_network.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                n_epi, score / print_interval, memory.size(), epsilon * 100))
            score = 0.0

    env.close()
