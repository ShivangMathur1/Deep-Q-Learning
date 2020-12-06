import gym
from Lunar_lander_agent import DQN
import torch as T
import numpy as np

if __name__ =='__main__':
    env = gym.make('LunarLander-v2')
    DQN = DQN(lr=0.001, inputDims=[8], fc1Dims=256, fc2Dims=256, nActions=4)
    DQN.load_state_dict(T.load('./lunar-model.pt'))

    scores = []
    epsHistory = []
    episodes = 500

    score = 0
    state = env.reset()
    done = False

    while not done:

        action = T.argmax(DQN(T.tensor([state]).to(DQN.device))).item()
        stateNew, reward, done, info = env.step(action)        
        env.render()
        score += reward
        state = stateNew

    print('Score: ', score)