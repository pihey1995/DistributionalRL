import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agents.q_learner import Q_learner
from agents.picnn_learner import picnn_learner
import matplotlib.pyplot as plt
from utils.utils import Normalizer
import os
import shutil
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

plt.rcdefaults()

args = dict()
args["BUFFER_SIZE"] = int(10000)  # replay buffer size
args["BATCH_SIZE"] = 256      # minibatch size
args["GAMMA"] = 0.99  # discount factor
args["TAU"] = 1e-1  # for soft update of target parameters #1
args["LR"] = 1e-3# learning rate
args["UPDATE_EVERY"] = 1  # how often to update the network
args["UPDATE_TARGET"] = 200
args["WARM_UP"] = 1000
args["ou_theta"] = 0.15
args["ou_sigma"] = 0.1
args["grad_norm"] = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


num_seeds = 1
seeds = [i for i in range(num_seeds)]

def icnn_dqn(agent, n_episodes=10000, max_t=1000, test_interval = 100, test_number = 1):
    actions_min, actions_max = agent.actions_range
    scores_train = []                        # list containing scores from each episode
    scores_test = []

    scores_window = deque(maxlen=100)  # last 100 scores

    normalizer = Normalizer(env.observation_space.shape[0])

    def normalize(normalizer, state):
        ##TODO: make this cleaner
        # normalizer.observe(state)
        return state  # normalizer.normalize(state)

    def run_tests(max_t,  test_number):
        score_test = 0
        for i in range(test_number):
            score_test_i = 0
            state = env.reset()
            state = normalize(normalizer, state)

            for t in range(max_t):
                action = agent.act(state, test = True).cpu().data.numpy()[0][0]
                next_state, reward, done, _ = env.step(action)
                next_state = normalize(normalizer, next_state)
                state = next_state
                score_test += reward
                if done:
                    break
            score_test+=score_test_i
        print(score_test)
        return score_test/float(test_number)

    def plot_q_distrib(state):
        actions_array = np.linspace(agent.actions_range[0], agent.actions_range[1], 100)
        actions = torch.Tensor(actions_array).to(device)
        actions = actions.unsqueeze(1)
        Q = agent.qnetwork_local.forward(observation=state.repeat(actions.size()[0], 1), actions = actions)["Q"].cpu().data.numpy().reshape(-1)
        plt.plot(actions_array, Q )
        plt.ylabel("Q")
        plt.xlabel("action")
        plt.show()

    for i_episode in range(1, n_episodes+1):

        state = env.reset()
        state = normalize(normalizer, state)

        score = 0
        if i_episode % 100000 == 0:
            state_0 = torch.Tensor([0 for i in range(agent.state_size)]).to(device)
            plot_q_distrib(state_0)

        for t in range(max_t):
            action = agent.act(state, test = False).cpu().data.numpy()[0][0]
            action = np.clip(action, a_min = actions_min, a_max = actions_max)

            next_state, reward, done, _ = env.step(action)
            next_state = normalize(normalizer, next_state)
            agent.step(state, action, reward, next_state, done)

            state = next_state
            score += reward
            if done:
                break


        if i_episode % test_interval == 0:
            score_test = run_tests(max_t, test_number)
            scores_test.append([i_episode, score_test])

        scores_window.append(score)       # save most recent score
        scores_train.append([i_episode, score])              # save most recent score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        #if np.mean(scores_window)>=200.0: ##To stop game if it is solved
        #    print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
        #    torch.save(agent.qnetwork_local.state_dict(), 'models/checkpoints/checkpoint.pth')
        #    break
    return scores_train, scores_test

for seed in seeds:
    #if(os.path.exists("./results/figs/seed-{}/".format(seed))):
    #    shutil.rmtree("./results/figs/seed-{}/".format(seed))
    #os.mkdir("./results/figs/seed-{}/".format(seed))

    env = gym.make('InvertedPendulum-v2')
    actions_range = env.action_space.low,env.action_space.high
    env.seed(seed=seed)
    agent = picnn_learner(state_size=env.observation_space.shape[0], action_size= env.action_space.shape[0], actions_range = actions_range, seed=seed, hiddens = [128,64], args = args)
    print(" ----------------- SEED:{}/{} -----------------".format(seed +1, len(seeds)))
    scores_train, scores_test = icnn_dqn(agent)
    #np.savetxt("./results/final_results2/train_score_seed_{}.csv".format(seed), np.array(scores_train), delimiter = ";")
    #np.savetxt("./results/final_results2/test_score_seed_{}.csv".format(seed), np.array(scores_test), delimiter = ";")
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()
#
