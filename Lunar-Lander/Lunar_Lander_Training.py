import os
import gym
from DQN_Agent import Agent
import torch as T
import matplotlib.pyplot as plt
import numpy as np

if __name__ =='__main__':
    env = gym.make('LunarLander-v2')
    brain = Agent(gamma=0.99, epsilon=1.0, batchSize=64, nActions=4, inputDims=[8], lr=0.001)

    scores = []
    episodes = 500
    learn = True
    quality = 0

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
            if learn:
                brain.learn()
                if i % 5 == 0 and i != 0:
                    brain.updateNetwork()

            state = stateNew
        
        scores.append(score)

        avgScore = np.mean(scores[-100:])
        print('Episode: ', i, '\tScore: ', score, '\tAverage Score: %.3f' % avgScore, 'Epsilon %.3f' % brain.epsilon)

        #Early stopping as soon as model performs successfully on 10 consecutive episodes
        if score >= 200:
            if learn:
                quality += 1
                learn = False
            else:
                quality += 1
                if quality > 10:
                    episodes = i + 1
                    break
        elif quality:
            learn = True
            quality = 0

    checkpoint = {'model': brain.DQN.state_dict(), 'score': scores, 'episodes': episodes}
    T.save(checkpoint, 'Lunar-Lander/lunar-model-dqn.pt')
    print('Model Saved')
    x = [i + 1 for i in range(episodes)]
    plt.plot(x, scores)
    plt.show()