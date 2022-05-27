import gym

import torch
import random
import collections

import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms

from network import DeepmindNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HyperParameter = collections.namedtuple('HyperParameter',
                                        ['batch_size', 'gamma', 'learning_rate', 'buffer_limit'])


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

        return torch.stack(state_list), torch.tensor(action_list, device=device), \
               torch.tensor(reward_list, device=device), torch.stack(state_prime_list), \
               torch.tensor(done_mask_list, device=device)

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
            self._policy_network = DeepmindNet()
            self._target_network = DeepmindNet()

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
            done=0 if done else 1
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
        super(Environment, self).__init__(gym.make('MsPacman-v0'))

        self._observation_queue = []

    def reset(self, **kwargs):
        state = super(Environment, self).reset(**kwargs)

        return self.observation(state)

    def step(self, action):

        state_prime, reward, done, info = super(Environment, self).step(action)

        state_prime = self.observation(state_prime)

        return state_prime, done, info

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


if __name__ == '__main__':

    # init part
    parameter = HyperParameter(
        batch_size=32,  # 한 학습당 사용되는 데이터 개수 (batch 크기)
        buffer_limit=100000,  # replay buffer 최대 사이즈
        gamma=0.98,  # 감쇄율 (Q-learning 포함)
        learning_rate=0.00025  # 학습률
    )

    env = Environment()
    agent = DQNAgent(param=parameter)

    # target network를 업데이트 하는 주기
    print_interval = 100
    score = 0.0

    for n_epi in range(100000):
        # e-greedy 행동 결정에 사용되는 epsilon, 에피소드가 지날 수록 값을 줄어나가게 함
        epsilon = max(0.1, 1 - n_epi / 100000)

        state = env.reset()
        done = False

        action_list = []
        episode_reward = 0

        while not done:
            # agent가 결정한 다음 행동
            action = agent.predict(state, epsilon)

            # 행동을 취한 후 결과로 S', R, 메타정보 등을 받음
            state_prime, reward, done, info = agent.step(env, state, action)

            state = state_prime

            score += reward

        # 에피소드가 끝날때마다 학습을 진행
        # 초기에는 학습 데이터가 적고, 학습 가치가 적으므로 충분히 Replay Buffer 에 쌓은 다음 학습 수행
        if agent._memory.size > 2000:
            agent.train()

        if n_epi % print_interval == 0 and n_epi != 0:
            # target network 업데이트
            agent.update_network()

            print("n_episode :{}, score : {:.1f}, eps : {:.1f}%".format(
                n_epi, score / print_interval, epsilon * 100))
            score = 0.0
