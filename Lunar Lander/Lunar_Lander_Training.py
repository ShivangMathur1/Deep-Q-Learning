import gym
from DQN_Agent import Agent
import torch as T
import matplotlib.pyplot as plt
import numpy as np

if __name__ =='__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=0.99, epsilon=1.0, batchSize=64, nActions=4, inputDims=[8], lr=0.001)

    scores = []
    epsHistory = []
    episodes = 500

    for i in range(episodes):
        score = 0
        state = env.reset()
        done = False

        while not done:
            action = brain.choose(state)
            stateNew, reward, done, info = env.step(action)
            
            if i % 10 == 0:
                env.render()

            score += reward
            brain.store(state, action, reward, stateNew, done)
            brain.learn()
            if i % 5 == 0 and i != 0:
                brain.updateNetwork()
            state = stateNew
        
        scores.append(score)
        epsHistory.append(brain.epsilon)

        avgScore = np.mean(scores[-100:])
        print('Episode: ', i, '\tScore: ', score, '\tAverage Score: %.3f' % avgScore, 'Epsilon %.3f' % brain.epsilon)

    T.save(brain.DQN.state_dict(), 'lunar-model.pt')

    x = [i + 1 for i in range(episodes)]
    plt.plot(x, scores)
    plt.show()