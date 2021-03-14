import gym
from DDQN_Agent import Agent
import torch as T
import matplotlib.pyplot as plt
import numpy as np

if __name__ =='__main__':
    env = gym.make('CartPole-v1')
    brain = Agent(gamma=0.99, epsilon=1.0, batchSize=16, nActions=2, inputDims=[4], lr=0.001, memSize=50000)

    scores = []
    epsHistory = []
    episodes = 200
    learn = True
    quality = 0

    for i in range(episodes):
        score = 0
        state = env.reset()
        done = False

        while not done:
            action = brain.choose(state)
            stateNew, reward, done, info = env.step(action)
            
            if i % 25 == 0:
                env.render()

            score += reward
            brain.store(state, action, reward, stateNew, done)
            if learn:
                brain.learn()
                if i % 5 == 0 and i != 0:
                    brain.updateNetwork()
            
            state = stateNew
            
        # Saving the model only if it has solved the environment
        if score >= 500:
            if learn:
                quality += 1
                learn = False
            else:
                quality += 1
                if quality > 5:
                    checkpoint = {'model': brain.DQN.state_dict(), 'score': scores, 'episodes': episodes}
                    T.save(checkpoint, 'CartPole/cartpole-model-ddqn.pt')
                    print('Model Saved')
                    episodes = i
                    break
        elif quality:
            learn = True
            quality = 0
        

        scores.append(score)
        epsHistory.append(brain.epsilon)

        avgScore = np.mean(scores[-50:])
        print('Episode: ', i, '\tScore: ', score, '\tAverage Score: %.3f' % avgScore, 'Epsilon %.3f' % brain.epsilon)

    x = [i + 1 for i in range(episodes)]
    plt.plot(x, scores)
    plt.show()