import numpy as np
import random
from collections import namedtuple, deque

from models.q_network import QNetwork
from models.distrib_q_network import Distrib_QNetwork

from utils.ReplayMemory import ReplayBuffer
import torch
import torch.nn.functional as F
import torch.optim as optim

#To check graph
#from torchviz import make_dot, make_dot_from_trace
#dot = make_dot(loss, params = dict(self.qnetwork_local.named_parameters()))
#dot.view()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Distrib_learner():

    def __init__(self, state_size, action_size, N, Vmin, Vmax,  hiddens, args, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed
        random.seed(seed)
        self.hiddens = hiddens
        self.BUFFER_SIZE = args["BUFFER_SIZE"]
        self.BATCH_SIZE = args["BATCH_SIZE"]
        self.GAMMA = args["GAMMA"]
        self.UPDATE_EVERY = args["UPDATE_EVERY"]
        self.UPDATE_TARGET = args["UPDATE_TARGET"]

        self.LR = args["LR"]
        self.TAU = args["TAU"]
        self.N = N
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.delta_z = (Vmax-Vmin)/(N-1)
        self.range_batch = torch.arange(self.BATCH_SIZE).long().to(device)
        self.qnetwork_local = Distrib_QNetwork(state_size, action_size , self.N, hiddens, seed).to(device)
        self.qnetwork_target = Distrib_QNetwork(state_size, action_size , self.N, hiddens, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)

        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed)
        self.t_step = 0
        #self.t_tot = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        #self.t_tot+=1
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        #self.update_target = (self.t_tot + 1) % self.UPDATE_TARGET
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)[0][0]
        self.qnetwork_local.eval()
        with torch.no_grad():
            z_dist = torch.from_numpy(
                np.array([[self.Vmin + i * self.delta_z for i in range(self.N)]])).to(device)
            z_dist = torch.unsqueeze(z_dist, 2).float()
            Q_dist, _  = self.qnetwork_local(state)#
            Q_dist = Q_dist.detach()
            Q_dist = Q_dist#.reshape(-1,  self.action_size, self.N)
            Q_target = torch.matmul(Q_dist, z_dist).squeeze(1)
            a_star = torch.argmax(Q_target, dim=1)[0]

        if eps != 0.:
            self.qnetwork_local.train()

        if random.random() > eps:
            return a_star.cpu().data.numpy()[0]
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        actions = actions.long()
        z_dist = torch.from_numpy(np.array([[self.Vmin + i*self.delta_z for i in range(self.N)]]*self.BATCH_SIZE)).to(device)
        z_dist = torch.unsqueeze(z_dist, 2).float()

        _, log_Q_dist_prediction = self.qnetwork_local(states)
        log_Q_dist_prediction = log_Q_dist_prediction[self.range_batch, actions.squeeze(1), :]#.reshape(-1, self.action_size, self.N)[self.range_batch, actions.squeeze(1), :]

        Q_dist_target, _ = self.qnetwork_target(next_states)
        Q_dist_target = Q_dist_target.detach()
        Q_dist_target = Q_dist_target#.reshape(-1, self.action_size, self.N)

        Q_target = torch.matmul(Q_dist_target, z_dist).squeeze(1)
        a_star = torch.argmax(Q_target, dim=1)
        Q_dist_star = Q_dist_target[self.range_batch, a_star.squeeze(1),:]

        m = torch.zeros(self.BATCH_SIZE,self.N).to(device)
        for j in range(self.N):
            T_zj = torch.clamp(rewards + self.GAMMA * (1-dones) * (self.Vmin + j*self.delta_z), min = self.Vmin, max = self.Vmax)
            bj = (T_zj - self.Vmin)/self.delta_z
            l = bj.floor().long()
            u = bj.ceil().long()

            mask_Q_l = torch.zeros(m.size()).to(device)
            mask_Q_l.scatter_(1, l, Q_dist_star[:,j].unsqueeze(1))
            mask_Q_u = torch.zeros(m.size()).to(device)
            mask_Q_u.scatter_(1, u, Q_dist_star[:,j].unsqueeze(1))
            m += mask_Q_l*(u.float() + (l == u).float()-bj.float())
            m += mask_Q_u*(-l.float()+bj.float())

        loss = - torch.sum(torch.sum(torch.mul(log_Q_dist_prediction, m),-1),-1) / self.BATCH_SIZE
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 5)

        self.optimizer.step()
        #if self.update_target == 0:
        #    self.hard_update(self.qnetwork_local, self.qnetwork_target)

        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)