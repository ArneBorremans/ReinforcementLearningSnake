from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, layers):
        # layers is for example [11, 256, 4] with 11 the input, 256 the hidden layer and 4 the output
        super().__init__()
        self.linears = []
        self.layers = layers

        # Input mapped to first hidden layer
        self.linears.append(nn.Linear(layers[0], layers[1]))

        print(self.linears)

        # Map all the hidden layers
        if len(layers) > 3:
            for index in range(1, len(layers)-2):
                self.linears.append(nn.Linear(layers[index], layers[index+1]))
                print(self.linears)

        # Map last hidden layer to output
        self.linears.append(nn.Linear(layers[-2], layers[-1]))
        print(self.linears)

        for index in range(1, len(self.linears)+1):
            self.add_module("linear" + str(index), self.linears[index-1])

    def forward(self, x):
        for index in range(0, len(self.linears)-1):
            x = F.relu(self.linears[index](x))

        x = self.linears[-1](x)
        return x

    def save(self, path='standard', file_name='model.pth'):
        model_folder_path = '../model/' + path

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new = R + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()