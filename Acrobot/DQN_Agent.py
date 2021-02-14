import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# Deep Q Network Class
class DQN(nn.Module):
    def __init__(self, lr, inputDims, fc1Dims, fc2Dims, nActions):
        super(DQN, self).__init__()
        
        print("input_dims ", inputDims[0], " n_actions ",nActions)
        self.fc1 = nn.Linear(*inputDims, fc1Dims)
        self.fc2 = nn.Linear(fc1Dims, fc2Dims)
        self.fc3 = nn.Linear(fc2Dims, nActions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        if T.cuda.is_available():
            print("Using CUDA")
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self.to(self.device)

    def forward(self, state):
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# Agent Class 
class Agent(object):
    def __init__(self, gamma, epsilon, lr, inputDims, batchSize, nActions, memSize=100000, epsilonFinal=0.05, epsilonDecrease=5e-4):
        # Brain of the agent
        self.DQN = DQN(lr, inputDims, 128, 128, nActions)
        self.DQNext = DQN(lr, inputDims, 128, 128, nActions)
        self.DQNext.load_state_dict(self.DQN.state_dict())
        
        # Hyper parameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilonFinal = epsilonFinal
        self.epsilonDecrease = epsilonDecrease
        self.lr = lr
        self.actionSpace = [i for i in range(nActions)]
        self.memSize = memSize
        self.batchSize = batchSize
        self.memCounter = 0
        self.replaceTarget = 100
        

        # Memory storage for sampling inputs
        self.stateMemory = np.zeros((memSize, *inputDims), dtype=np.float32)
        self.newStateMemory = np.zeros((memSize, *inputDims), dtype=np.float32)        
        self.actionMemory = np.zeros(memSize, dtype=np.int32)
        self.rewardMemory = np.zeros(memSize, dtype=np.float32)
        self.terminalMemory = np.zeros(memSize, dtype=np.bool)

    # Storage of state, action, reward and termination values
    def store(self, state, action, reward, newState, terminal):
        index = self.memCounter % self.memSize
        self.stateMemory[index] = state
        self.newStateMemory[index] = newState
        self.rewardMemory[index] = reward
        self.actionMemory[index] = action
        self.terminalMemory[index] = terminal
        
        self.memCounter += 1

    # Exploration vs Exploitation
    def choose(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.actionSpace)
        else:
            state = T.tensor([observation]).to(self.DQN.device)
            actions = self.DQN(state.float())
            action = T.argmax(actions).item()

        return action

    # Actual learning
    def learn(self):
        # If no batch is available, then don't learn
        if self.memCounter < self.batchSize:
            return
        # Start by initializing the gradients
        self.DQN.optimizer.zero_grad()
        
        # Generate batch
        maxMem = min(self.memCounter, self.memSize)
        batch = np.random.choice(maxMem, self .batchSize, replace=False)
        batchIndex = np.arange(self.batchSize, dtype=np.int32)

        stateBatch = T.tensor(self.stateMemory[batch]).to(self.DQN.device)
        newStateBatch = T.tensor(self.newStateMemory[batch]).to(self.DQN.device)
        rewardBatch = T.tensor(self.rewardMemory[batch]).to(self.DQN.device)
        terminalBatch = T.tensor(self.terminalMemory[batch]).to(self.DQN.device)
        actionBatch = self.actionMemory[batch]  

        # Forward DQN for this and the next state
        # Update target Q values of the whole batch
        # Qtarget = reward + gamma*(q-value_for_best_action)
        # We get index 0 as the max function returns a tuple (value, index)
        qEval = self.DQN(stateBatch)[batchIndex, actionBatch]
        qNext = self.DQNext(newStateBatch)
        qNext[terminalBatch] = 0.0
        qTarget = rewardBatch + self.gamma * T.max(qNext, dim=1)[0]

        # Backpropagate the loss and Optimize
        loss = self.DQN.loss(qTarget, qEval).to(self.DQN.device)
        loss.backward();
        self.DQN.optimizer.step()
        self.updateEpsilon()

    # Update exploration chance
    def updateEpsilon(self):
        self.epsilon = self.epsilon - self.epsilonDecrease if self.epsilon > self.epsilonFinal else self.epsilonFinal
    
    def updateNetwork(self):
        self.DQNext.load_state_dict(self.DQN.state_dict())