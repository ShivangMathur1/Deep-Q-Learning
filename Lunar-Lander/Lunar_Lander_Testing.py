import gym
from DQN_Agent import DQN
import torch as T

if __name__ =='__main__':
    episodes = 100
    env = gym.make('LunarLander-v2')
    DQN = DQN(lr=0.001, inputDims=[8], fc1Dims=256, fc2Dims=256, nActions=4)
    DQN.load_state_dict(T.load('Lunar-Lander/lunar-model-ddqn.pt')['model'])

    scores = []

    for episode in range(episodes):
        score = 0
        state = env.reset()
        done = False

        while not done:

            action = T.argmax(DQN(T.tensor([state]).to(DQN.device))).item()
            stateNew, reward, done, info = env.step(action)        
            # env.render()
            score += reward
            state = stateNew
        scores.append(score)
        print(episode + 1, 'Score: ', score)
    print('Average score: ', sum(scores)/episodes)