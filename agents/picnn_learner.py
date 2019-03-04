import numpy as np
import random
from collections import namedtuple, deque

from models.q_network import QNetwork
from models.picnn_network import picnn_network

from utils.ReplayMemory import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class picnn_learner():

    def __init__(self, state_size, action_size, actions_range, hiddens, args, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.actions_range = actions_range
        self.seed = seed
        random.seed(seed)

        self.hiddens = hiddens
        self.ou_theta = args["ou_theta"]
        self.ou_sigma = args["ou_sigma"]
        self.BUFFER_SIZE = args["BUFFER_SIZE"]
        self.BATCH_SIZE = args["BATCH_SIZE"]
        self.GAMMA = args["GAMMA"]
        self.UPDATE_EVERY = args["UPDATE_EVERY"]
        self.WARM_UP = args["WARM_UP"]
        self.LR = args["LR"]
        self.TAU = args["TAU"]
        self.grad_norm = args["grad_norm"]

        self.qnetwork_local = picnn_network(input_shape = state_size, action_shape = action_size, actions_range = actions_range,  hiddens = hiddens, seed =  seed).to(device)
        self.qnetwork_target = picnn_network(input_shape = state_size, action_shape = action_size, actions_range = actions_range,  hiddens = hiddens, seed =  seed).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            if len(self.memory) > self.WARM_UP:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, test = False):
        actions_min, actions_max = self.actions_range
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        #self.qnetwork_local.eval()
        #with torch.no_grad():
        a_star = self.qnetwork_local.best_action(state)["actions"]#.cpu().data.numpy()[0]

        if not test:
            #self.qnetwork_local.train()
            x = getattr(self, "noise", a_star.clone().zero_())
            mu = a_star.clone().zero_()
            dx = self.ou_theta * (mu-x) + self.ou_sigma * x.clone().normal_()
            self.noise = x + dx
            a_star += self.noise

        return a_star

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions = self.qnetwork_local.best_action(next_states)["actions"]
        max_Q = self.qnetwork_target.forward(observation=next_states, actions=next_actions, entropy=True)["Q"][0][0].detach()
        targets = rewards + (1 - dones) * self.GAMMA * max_Q
        predictions = self.qnetwork_local.forward(observation=states, actions=actions, entropy=True)["Q"][0][0]
        loss = torch.mean((targets-predictions)**2)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), self.grad_norm)

        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)