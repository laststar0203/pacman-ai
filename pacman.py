from dqn import *

SAVE_MODEL_DIR = "/"
SAVE_MODEL_PREFIX = "Pacman-ai-"

parameter = HyperParameter(
    batch_size=32,  # 한 학습당 사용되는 데이터 개수 (batch 크기)
    buffer_limit=100000,  # replay buffer 최대 사이즈
    gamma=0.98,  # 감쇄율 (Q-learning 포함)
    learning_rate=0.00025,  # 학습률
    epoch=10,
    update_period=100,
)

if __name__ == '__main__':

    env = Environment()
    agent = DQNAgent(param=parameter)

    # target network를 업데이트 하는 주기
    score = 0.0

    n_epi = 0

    while True:

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
            state_prime, reward, done, info = env.step(action)

            agent.memorize(state, state_prime, action, reward)

            state = state_prime

            score += reward

        # 에피소드가 끝날때마다 학습을 진행
        # 초기에는 학습 데이터가 적고, 학습 가치가 적으므로 충분히 Replay Buffer 에 쌓은 다음 학습 수행
        if n_epi > 3:
            agent.train()

        if n_epi % 100 == 0 and n_epi != 0:
            print("n_episode :{}, score : {:.1f}, eps : {:.1f}%".format(n_epi, score / 100, epsilon * 100))
            score = 0.0

        if n_epi % 1000:
            agent.save(f"{SAVE_MODEL_PREFIX}{SAVE_MODEL_DIR}{n_epi}")

        n_epi += 1