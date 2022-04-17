import gym
import time

env = gym.make('MsPacman-v0')
state = env.reset()

observation = None
reward = None
done = None
info = None

for _ in range(100000):
    screen = env.render()

    observation, reward, done, info = env.step(6)

    time.sleep(0.001)

    if done:
        env.reset()
env.close()
