import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ReplayBuffer:
    def __init__(self):
        self.buffer = collections.deque()
        self.batch_size = 32
        self.size_limit = 50000

    def put(self, data):
        self.buffer.append(data)
        if len(self.buffer) > self.size_limit:
            self.buffer.popleft()

    def sample(self, n):
        return random.sample(self.buffer, n)

    def size(self):
        return len(self.buffer)


class Qnet(nn.Module):

    def __init__(self):
        super(Qnet, self).__init__()

        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        print(x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)
        coin = random.random()

        if coin < epsilon:
            return random.randint(0, 1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, gamma, optimizer, batch_size):
    # 한번만 해도되지만 작성자 맘으로 했다고 함
    for i in range(10):
        batch = memory.sample(batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        # 1. 리스트화 함
        for transition in batch:
            # s는 상태, a는 액션, r은 리워드, s_prime은 다음 상태
            s, a, r, s_prime, done_mask = transition

            s_lst.append(s)
            # s와 차원을 맞추기 위함
            a_lst.append([a])
            r_lst.append([r])

            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        # 2. tensor로 만듬
        s, a, r, s_prime, done_mask = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                      torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                      torch.tensor(done_mask_lst)

        # input: s.shape([32,4]) output: s.shape([32,2]) gpu적으로 성능ㅇ ㅣ좋다고 함
        q_out = q(s)
        q_a = q_out.gather(1, a)  # 취한 action의 q값만 골라냄, 1번째 차원에서 골른다. [32,2 <-]
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # 차원 맞추기
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main():
    env = gym.make('CartPole-v1')
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    avg_t = 0
    gamma = 0.98
    batch_size = 32
    # q만 옵티마이징 q_target는 옵티마이징 안함.
    optimizer = optim.Adam(q.parameters(), lr=0.0005)

    render = False

    # 에피소드 10000개
    for n_epi in range(10000):
        # 에피소드가 올라갈수록 8%에서 1%로 줄임
        epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))
        s = env.reset()
        print(s)
        for t in range(600):
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            print(s)
            s_prime, r, done, info = env.step(a)
            print(type(s))

            # done_mask는 곱해지는 값이다 게임이 끝나지 않으면 0으로 결과가 나오게 할려고 함
            done_mask = 0.0 if done else 1.0

            memory.put((s, a, r / 100.0, s_prime, done_mask))
            s = s_prime

            if render:
                env.render()
                import time
                time.sleep(0.01)

            if done:
                break

            import time
            time.sleep(1)

        avg_t += t

        if avg_t / 20.0 > 300:
            render = True

        # 충분히 쌓인 다음에 시작
        if memory.size() > 2000:
            train(q, q_target, memory, gamma, optimizer, batch_size)

        if n_epi % 20 == 0 and n_epi != 0:
            q_target.load_state_dict(q.state_dict())
            print("# of episode: {}, Avg timestep : {:.1f}, buffer size : {}, epsilon : {:.1f}%".format(n_epi,
                                                                                                        avg_t / 20.0,
                                                                                                        memory.size(),
                                                                                                        epsilon * 100))

            avg_t = 0


if __name__ == '__main__':
    main()
