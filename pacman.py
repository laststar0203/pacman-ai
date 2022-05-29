import time
import pandas as pd
from dqn import *

SAVE_MODEL_DIR = "./model/20220529"
SAVE_MODEL_PREFIX = "Pacman-ai-"

parameter = HyperParameter(
    batch_size=32,  # 한 학습당 사용되는 데이터 개수 (batch 크기)
    buffer_limit=100000,  # replay buffer 최대 사이즈
    gamma=0.98,  # 감쇄율 (Q-learning 포함)
    learning_rate=0.00025,  # 학습률
    epoch=100,
    update_period=100,
)

# global variable

agent = None
savefile = None
render = False

csv_interval = 100
save_interval = 1000

result = []

if __name__ == '__main__':

    env = Environment()
    agent = DQNAgent(param=parameter)

    n_epi = 0

    while True:

        # e-greedy 행동 결정에 사용되는 epsilon, 에피소드가 지날 수록 값을 줄어나가게 함
        epsilon = max(0.1, 1 - n_epi / 100000)

        state = env.reset()
        done = False

        action_list = []
        episode_reward = 0
        loss = -1

        while not done:
            # agent가 결정한 다음 행동
            action = agent.predict(state, epsilon)

            # 행동을 취한 후 결과로 S', R, 메타정보 등을 받음
            state_prime, reward, done, info = env.step(action)

            agent.memorize(state, state_prime, action, reward)

            state = state_prime

            episode_reward += reward

            if render:
                env.render()
                time.sleep(0.01)

        # 에피소드가 끝날때마다 학습을 진행
        # 초기에는 학습 데이터가 적고, 학습 가치가 적으므로 충분히 Replay Buffer 에 쌓은 다음 학습 수행
        if n_epi > 3:
            loss = agent.train().mean()

        if n_epi % csv_interval == 0 and n_epi != 0:
            pd.DataFrame(result).to_csv(f'{SAVE_MODEL_DIR}/result.csv')

        if n_epi % save_interval == 0 and n_epi != 0:
            agent.save(f"{SAVE_MODEL_DIR}/{SAVE_MODEL_PREFIX}{n_epi}.pi")
            score = 0.0

        result.append({
            'loss': loss,
            'reward': episode_reward,
            'epsilon': epsilon
        })

        n_epi += 1
