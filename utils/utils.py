import numpy as np

def smooth_update(policy_net, target_net, tau):
    model_policy_net = policy_net.state_dict()
    model_target_net = target_net.state_dict()
    for param in model_target_net.keys():
        model_target_net[param] = (1-tau) * model_target_net[param] + tau * model_policy_net[param]
    target_net.load_state_dict(model_target_net)
    return

class Normalizer():
    def __init__(self, num_inputs):
        self.n = np.zeros(num_inputs)
        self.mean = np.zeros(num_inputs)
        self.mean_diff = np.zeros(num_inputs)
        self.var = np.zeros(num_inputs)

    def observe(self, x):
        self.n += 1.
        last_mean = self.mean
        self.mean += (x-self.mean)/self.n
        self.mean_diff += (x-last_mean)*(x-self.mean)
        self.var = np.clip(self.mean_diff/self.n, a_min=1e-2, a_max = np.infty)

    def normalize(self, input):
        obs_std = np.sqrt(self.var)
        return (input - self.mean)/obs_std