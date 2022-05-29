import gym

import torch
import random
import collections

import numpy as np

import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

from network import DeepmindNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HyperParameter = collections.namedtuple('HyperParameter',
                                        ['batch_size', 'gamma', 'learning_rate', 'buffer_limit', 'epoch',
                                         'update_period'])


class ReplayBuffer:

    def __init__(self, buffer_limit):
        self._buffer = collections.deque(maxlen=buffer_limit)

    @property
    def size(self):
        return len(self._buffer)

    def put(self, state, state_prime, action, reward):
        self._buffer.append((state, state_prime, action, reward))

    def sample(self, n):
        mini_batch = random.sample(self._buffer, n)
        state_list, action_list, reward_list, state_prime_list = [], [], [], []

        for transition in mini_batch:
            state, state_prime, action, reward = transition

            state_list.append(state)
            action_list.append([action])
            reward_list.append([reward])
            state_prime_list.append(state_prime)

        return torch.stack(state_list), torch.tensor(action_list, device=device), \
               torch.tensor(reward_list, device=device), torch.stack(state_prime_list)

    def reset(self):
        self._buffer.clear()


class DQNAgent:

    def __init__(self, param: HyperParameter, path=None):
        self._PARAMETER = param
        self._policy_network = None
        self._target_network = None

        self._memory = ReplayBuffer(param.buffer_limit)

        # 파일시스템에 저장된 모델을 로드
        if path:
            self.load(path)
        else:
            self._policy_network = DeepmindNet()
            self._target_network = DeepmindNet()

            self.update_network()

        self._optimizer = optim.Adam(self._policy_network.parameters(), lr=param.learning_rate)

        self._train_count = 0

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

    def memorize(self, state, state_prime, action, reward):
        self._memory.put(
            state=state,
            state_prime=state_prime,
            action=action,
            reward=reward
        )

    def train(self):

        loss_list = []

        # 학습 데이터를 랜덤으로 추출
        state_list, action_list, reward_list, state_prime_list = self._memory.sample(self._PARAMETER.batch_size)

        for _ in range(self._PARAMETER.epoch):

            # 행동 정책 q-value
            output = self._policy_network(state_list)
            q_action = output.gather(1, action_list)

            # 타켓 정책 q-value
            max_q_prime = self._target_network(state_prime_list).max(1)[0].unsqueeze(1)
            target = reward_list + self._PARAMETER.gamma * max_q_prime

            # 역전파 연산
            loss = F.smooth_l1_loss(q_action, target)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            loss_list.append(float(loss))

        if self._train_count % self._PARAMETER.update_period == 0:
            self.update_network()

        self._train_count += 1

        return np.array(loss_list)

    def save(self, path):
        """ 모델을 파일시스템에 저장 """
        torch.save(self._target_network.state_dict(), path)

    def load(self, path):
        """ 파일시스템에 저장되어있는 모델을 불러옴 """
        self._policy_network = DeepmindNet()
        self._target_network = DeepmindNet()

        self._policy_network.load_state_dict(torch.load(path))
        self._policy_network.eval()

        self._memory.reset()

        self.update_network()


class Environment(gym.Wrapper):

    def __init__(self):
        super(Environment, self).__init__(gym.make('ALE/MsPacman-v5'))

        self._observation_queue = []

    def reset(self, **kwargs):
        state = super(Environment, self).reset(**kwargs)

        return self.observation(state)

    def step(self, action):

        state_prime, reward, done, info = super(Environment, self).step(action)

        state_prime = self.observation(state_prime)

        return state_prime, reward, done, info

    def observation(self, observation):
        """
        MsPacman-v0 는 기본으로 이미지 형태로 제공 shape=(210, 160, 3),
        이를 컬러를 지운 3 프레임으로 구성된 이미지 형태로 전처리 한다. shape=(3, 84, 84)
         """

        # 컬러를 지운후 84X84 이미지로 변환 후 학습을 위해 tensor 로 변환
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((110, 84)),
            transforms.ToTensor()
        ])

        current = transform(observation)[:84, :84]

        """
        저장해둔 이전 상태 이미지 2개를 병합하여 3 프레임으로 구성된 state 값을 만듬

        현재 상태 = C 일때
        A -> B -> C 순으로 상태가 변화했다 가정하면

        (큐 크기가 0 ~ 1개 일 경우)
            [    ] -> C C C
            [B   ] -> C C C
        (큐 크기가 2 개 일 경우)
            [A, B] -> A B C

        """

        frame = [None, None, None]

        if len(self._observation_queue) < 2:
            frame[0] = current
            frame[1] = current
        else:
            frame[0] = self._observation_queue.pop(0)
            frame[1] = self._observation_queue[0]

        frame[2] = current

        self._observation_queue.append(current)

        return torch.cat(frame)
