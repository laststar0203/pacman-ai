import gym

import random
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HyperParameter = collections.namedtuple('HyperParameter',
                                        ['batch_size', 'gamma', 'learning_rate', 'buffer_limit'])


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()
        # self._conv1 = nn.Conv2d(in_channels=3,
        #                         out_channels=16,
        #                         kernel_size=8,
        #                         stride=4,
        #                         device=device)
        #
        # self._bn1 = nn.BatchNorm2d(16, device=device)
        #
        # self._conv2 = nn.Conv2d(in_channels=16,
        #                         out_channels=32,
        #                         kernel_size=4,
        #                         stride=2,
        #                         device=device)
        #
        # self._bn2 = nn.BatchNorm2d(32,
        #                            device=device)
        #
        # self._ln1 = nn.Linear(2592, 256,
        #                       device=device)
        # self._ln2 = nn.Linear(256, 9,
        #                       device=device)

        self._conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=2,
                device=device),
            nn.BatchNorm2d(16, device=device),
            nn.ReLU()
        )

        self._mp1 = nn.MaxPool2d(kernel_size=2)

        self._conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=2,
                      device=device),
            nn.BatchNorm2d(32, device=device),
            nn.ReLU()
        )

        self._mp2 = nn.MaxPool2d(kernel_size=3)

        self._ln1 = nn.Sequential(
            nn.Linear(32768, 1024, device=device),
            nn.ReLU()
        )

        self._ln2 = nn.Sequential(
            nn.Linear(1024, 256, device=device),
            nn.ReLU()
        )

        self._ln3 = nn.Linear(256, 9,
                              device=device)

    def forward(self, x):
        x = x.to(device)
        x = self._conv1(x)
        x = self._mp1(x)

        x = self._conv2(x)
        x = self._mp2(x)

        x = x.view(-1) if x.dim() == 3 else x.view(x.shape[0], -1)

        x = self._ln1(x)
        x = self._ln2(x)
        x = self._ln3(x)

        return x


class ReplayBuffer:

    def __init__(self, buffer_limit):
        self._buffer = collections.deque(maxlen=buffer_limit)

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

    def reset(self):
        self._buffer.clear()


class DQNAgent:

    def __init__(self, param: HyperParameter, path=None):
        self._PARAMETER = param
        self._policy_network = None
        self._target_network = None

        self._memory = ReplayBuffer(param.buffer_limit)

        if path:
            self.load(path)
        else:
            self._policy_network = Qnet()
            self._target_network = Qnet()

            self.update_network()

        self._optimizer = optim.Adam(self._policy_network.parameters(), lr=param.learning_rate)

    def update_network(self):
        self._target_network.load_state_dict(self._policy_network.state_dict())

    def predict(self, state, epsilon):

        out = self._policy_network(state.unsqueeze(0))

        r = random.random()

        # epsilon greedy
        if r < epsilon:
            return random.randint(0, 8)
        else:
            return out.argmax().item()

    def step(self, env, state, action):

        state_prime, reward, done, info = env.step(action)

        self._memory.put(
            state=state,
            state_prime=state_prime,
            action=action,
            reward=reward,
            done=done
        )

        return state_prime, reward, done, info

    def train(self):
        state_list, action_list, reward_list, state_prime_list, \
        done_mask_list = self._memory.sample(self._PARAMETER.batch_size)

        output = self._policy_network(state_list)
        q_action = output.gather(1, action_list)

        max_q_prime = self._target_network(state_prime_list).max(1)[0].unsqueeze(1)
        target = reward_list + self._PARAMETER.gamma * max_q_prime * done_mask_list

        loss = F.smooth_l1_loss(q_action, target)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def save(self, path):
        torch.save(self._policy_network.state_dict(), path)

    def load(self, path):
        self._policy_network = Qnet()
        self._target_network = Qnet()

        self._policy_network.load_state_dict(torch.load(path))
        self._policy_network.eval()

        self._memory.reset()

        self.update_network()


class Environment(gym.Wrapper):
    move = -1
    eat_cookie = 10
    eat_ghost = 0
    death = -2

    def __init__(self):
        super(Environment, self).__init__(gym.make('MsPacman-v0'))

        self._move_reward = Environment.move
        self._eat_cookie_reward = Environment.eat_cookie
        self._death_reward = Environment.death
        self._eat_ghost_reward = Environment.eat_ghost

    def reset(self,
              move_reward: int = move,
              eat_cookie_reward: int = eat_cookie,
              death_reward: int = death,
              eat_ghost_reward: int = eat_ghost,
              **kwargs):

        self._move_reward = move_reward
        self._eat_cookie_reward = eat_cookie_reward
        self._death_reward = death_reward
        self._eat_ghost_reward = eat_ghost_reward

        self._metadata = {
            'lives': 3,
            'get_coin': 0
        }

        state = super(Environment, self).reset(**kwargs)
        return self.observation(state)

    def step(self, action):

        state_prime, reward, done, info = super(Environment, self).step(action)

        state_prime = self.observation(state_prime)
        new_reward = self.reward(reward, info)

        self._metadata['lives'] = info['lives']

        return state_prime, new_reward, done, info

    def reward(self, reward, info):

        new_reward = 0

        # move
        if reward == 0:
            new_reward = self._move_reward
        # eat
        elif reward == 10:
            new_reward = self._eat_cookie_reward
            # self._metadata['get_coin'] += 1
        else:
            new_reward = 0

        if self._metadata['lives'] > info['lives'] != 3:
            # new_reward += self._death_reward * (1 - (self._metadata['get_coin'] % 150) / 150)
            new_reward += self._death_reward

        return new_reward

    def observation(self, observation):
        observation = observation[1:172, 1:160]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((200, 200)),
            transforms.ToTensor()
        ])

        return transform(observation)


if __name__ == '__main__':

    parameter = HyperParameter(
        batch_size=32,
        buffer_limit=50000,
        gamma=0.98,
        learning_rate=0.001
    )

    env = Environment()
    agent = DQNAgent(param=parameter)

    print_interval = 20
    score = 0.0

    for n_epi in range(1000):
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))

        state = env.reset()
        done = False

        while not done:
            action = agent.predict(state, epsilon)
            state_prime, reward, done, info = agent.step(env, state, action)

            state = state_prime

            score += reward

        if n_epi % print_interval == 0 and n_epi != 0:
            agent.update_network()

            print("n_episode :{}, score : {:.1f}, eps : {:.1f}%".format(
                n_epi, score / print_interval, epsilon * 100))
            score = 0.0
