import torch as th
from torch import nn
import torch.nn.functional as F
from functools import partial
from itertools import chain
import numpy as np
from torch.optim import SGD
from utils.icnn_utils.utils import proj_newton
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
import time

class HyperLinear(nn.Module):
    """
    Allow partial weight constraints needed for ICNN
    Linear network layers that allows for two additional complications:
        - parameters admit to be connected via a hyper-network like structure
        - network weights are transformed according to some rule before application
    """

    def __init__(self, in_size, out_size, y_size=None, use_y=False, use_hypernetwork=False, use_bias=False):
        super(HyperLinear, self).__init__()

        self.use_hypernetwork = use_hypernetwork
        self.use_y = use_y
        self.use_bias = use_bias
        if not self.use_hypernetwork:
            self.w_z = nn.Linear(in_size, out_size, bias=self.use_bias)

        stdv_z = 1. / np.sqrt(in_size)

        if not self.use_bias:
            self.b = nn.Parameter(th.randn(out_size))
            self.b.data.uniform_(-stdv_z, stdv_z)

        # initialize layers

        if not self.use_hypernetwork:
            self.w_z.weight.data.uniform_(-stdv_z, stdv_z)

        if self.use_y:
            stdv_y = 1. / np.sqrt(y_size)
            self.w_y = nn.Linear(y_size, out_size, bias=self.use_bias)
            self.w_y.weight.data.uniform_(-stdv_y, stdv_y)
        pass

    def forward(self, inputs, weights=None, y=None, weight_mod="abs", hypernet=None, **kwargs):

        assert inputs.dim() == 2, "we require inputs to be of shape [a*bs*t]*v"

        if self.use_hypernetwork:
            assert weights is not None, "if using hyper-network, need to supply the weights!"
            w_z = weights
        else:
            w_z = self.w_z.weight  # .weight.data
        if self.use_y:
            w_y = self.w_y.weight  # .weight.data

        weight_mod_fn = None
        if weight_mod in ["abs"]:
            weight_mod_fn = th.abs
        elif weight_mod in ["pow"]:
            exponent = kwargs.get("exponent", 2)
            weight_mod_fn = partial(th.pow, exponent=exponent)
        elif callable(weight_mod):
            weight_mod_fn = weight_mod

        if weight_mod_fn is not None:
            w_z = weight_mod_fn(w_z)

        if not self.use_y:
            x = th.mm(inputs, w_z.t())
        else:
            x = th.mm(inputs, w_z.t()) + th.mm(y, w_y.t())

        if self.use_bias:
            x = x + self.b

        return x

class picnn_network(nn.Module):
    def __init__(self, input_shape, action_shape, actions_range, hiddens, seed, batch_norm = False):
        super(picnn_network, self).__init__()

        self.n_hiddens_u_path = hiddens
        self.n_hiddens_z_path = hiddens
        self.n_actions = action_shape
        self.actions_range = actions_range
        self.observation_size = input_shape
        self.batch_norm = batch_norm
        self.bn_init = nn.BatchNorm1d(self.observation_size)
        self.seed = seed

        if self.observation_size != 0:
            for i in range(len(self.n_hiddens_u_path)):
                if i == 0:
                    self.w_utilde0 = nn.Linear(self.observation_size, self.n_hiddens_u_path[i])
                    std_w0 = 1/np.sqrt(self.observation_size)
                    self.w_utilde0.weight.data.uniform_(-std_w0, std_w0)

                    if self.batch_norm:
                        self.bn0 = nn.BatchNorm1d(self.n_hiddens_u_path[i])

                    self.w_utilde0.bias.data.uniform_(0, 0)

                    self.w_zu0 = nn.Linear(self.observation_size, self.n_actions)
                    self.w_zu0.weight.data.uniform_(-std_w0, std_w0)
                    self.w_zu0.bias.data.uniform_(1, 1)

                    self.w_u0 = nn.Linear(self.observation_size, self.n_hiddens_z_path[i])
                    self.w_u0.weight.data.uniform_(-std_w0, std_w0)
                    self.w_u0.bias.data.uniform_(0, 0)

                    self.w_yu0 = nn.Linear(self.observation_size, self.n_actions)
                    self.w_yu0.weight.data.uniform_(- std_w0, std_w0)
                    self.w_yu0.bias.data.uniform_(1,1)

                else:
                    std_w_i = 1/np.sqrt(self.n_hiddens_u_path[i-1])


                    setattr(self, "w_utilde{}".format(i), nn.Linear(self.n_hiddens_u_path[i - 1], self.n_hiddens_u_path[i]))
                    getattr(self, "w_utilde{}".format(i)).weight.data.uniform_(-std_w_i, std_w_i)
                    if self.batch_norm:
                        setattr(self, "bn{}".format(i), nn.BatchNorm1d(self.n_hiddens_u_path[i]))


                    getattr(self, "w_utilde{}".format(i)).bias.data.uniform_(0, 0)

                    setattr(self, "w_zu{}".format(i), nn.Linear(self.n_hiddens_u_path[i - 1], self.n_hiddens_z_path[i-1]))
                    getattr(self, "w_zu{}".format(i)).weight.data.uniform_(-std_w_i, std_w_i)
                    getattr(self, "w_zu{}".format(i)).bias.data.uniform_(1, 1)

                    setattr(self, "w_u{}".format(i), nn.Linear(self.n_hiddens_u_path[i - 1], self.n_hiddens_z_path[i]))
                    getattr(self, "w_u{}".format(i)).weight.data.uniform_(-std_w_i, std_w_i)
                    getattr(self, "w_u{}".format(i)).bias.data.uniform_(0, 0)

                    setattr(self, "w_yu{}".format(i), nn.Linear(self.n_hiddens_u_path[i - 1], self.n_actions))
                    getattr(self, "w_yu{}".format(i)).weight.data.uniform_(-std_w_i, std_w_i)
                    getattr(self, "w_yu{}".format(i)).bias.data.uniform_(1,1)

            l = len(self.n_hiddens_u_path)
            std_w_l = 1 / np.sqrt(self.n_hiddens_u_path[l - 1])

            setattr(self, "w_zu{}".format(l), nn.Linear(self.n_hiddens_u_path[l - 1], self.n_hiddens_z_path[l-1]))
            getattr(self, "w_zu{}".format(l)).weight.data.uniform_(-std_w_l, std_w_l)
            getattr(self, "w_zu{}".format(l)).bias.data.uniform_(1, 1)

            setattr(self, "w_u{}".format(l), nn.Linear(self.n_hiddens_u_path[l - 1], 1))
            getattr(self, "w_u{}".format(l)).weight.data.uniform_(-std_w_l, std_w_l)
            getattr(self, "w_u{}".format(l)).bias.data.uniform_(0, 0)

            setattr(self, "w_yu{}".format(l), nn.Linear(self.n_hiddens_u_path[l - 1], self.n_actions))
            getattr(self, "w_yu{}".format(l)).weight.data.uniform_(-std_w_l, std_w_l)
            getattr(self, "w_yu{}".format(l)).bias.data.uniform_(1, 1)

            pass


        ##z path
        for i, n_hidden in enumerate(self.n_hiddens_z_path):
            if i == 0:
                self.fc0 = HyperLinear(self.n_actions, self.n_hiddens_z_path[0])
            else:
                setattr(self, "fc{}".format(i),
                        HyperLinear(self.n_hiddens_z_path[i - 1], n_hidden, y_size=self.n_actions, use_y=True))
                setattr(self, "fc{}".format(len(self.n_hiddens_z_path)),
                        HyperLinear(self.n_hiddens_z_path[-1], 1, y_size=self.n_actions, use_y=True))

        pass

    def forward(self, observation, actions=None, entropy=False):

        if len(observation.size()) == 2:
            assert observation.size()[1] == self.observation_size

        # elif len(observation.size()) == 1:
        #     observation = th.unsqueeze(observation, 0)
        #
        # if len(actions.size()) != 2:
        #     actions = actions.contiguous().view(-1, actions.shape[-1])

        if self.observation_size == 0:
            for i in range(len(self.n_hiddens_z_path)):
                if i == 0:
                    q = F.relu(getattr(self, "fc{}".format(i))(inputs=actions))
                else:
                    q = F.relu(getattr(self, "fc{}".format(i))(inputs=q, y=actions))
                    Q = - getattr(self, "fc{}".format(len(self.n_hiddens_z_path)))(inputs=q, y=actions)

        else:

            prev_u = observation.clone()
            if prev_u.size()[0] != 1 and self.batch_norm:
                prev_u = self.bn_init(input=prev_u)
            action_clone = actions.clone().to(device)

            prev_z = actions.clone().to(device)# th.ones((action_clone.size()), device=prev_u.device)
            for i in range(len(self.n_hiddens_z_path)+1):

                z_ = prev_z * F.relu(getattr(self, "w_zu{}".format(i))(input=prev_u))
                y_ = action_clone * getattr(self, "w_yu{}".format(i))(input=prev_u)
                u_ = getattr(self, "w_u{}".format(i))(input=prev_u)
                if i != len(self.n_hiddens_z_path):
                    u = F.relu(getattr(self, "w_utilde{}".format(i))(input=prev_u))
                    if u.size()[0] != 1 and self.batch_norm:
                        u = getattr(self, "bn{}".format(i))(input=u)
                    z = F.leaky_relu(u_ + getattr(self, "fc{}".format(i))(inputs=z_, y=y_), negative_slope=0.01)
                else:
                    z = u_ + getattr(self, "fc{}".format(i))(inputs=z_, y=y_)
                prev_u, prev_z = u, z
            Q = - z

            pass

        if entropy:
            action_min, action_max = self.actions_range
            # for _aid in range(self.n_agents):
            #     for _actid in range(self.args.action_spaces[_aid].shape[0]):
            #         pass
            for i in range(self.n_actions):
                action_01_i = (actions - action_min[i]) / (action_max[i] - action_min[i])
                Q += -(action_01_i * th.log(action_01_i + 1e-5) + (1 - action_01_i) * th.log(1 - action_01_i +  + 1e-5))

        return {"Q": Q,
                }

    def get_gradient_batch(self, observation, actions):
        actions = actions.detach()
        actions.requires_grad = True
        optimizer_action = SGD([actions], lr=0.01)
        optimizer_action.zero_grad()
        Q = self.forward(observation=observation, actions=actions)["Q"]
        Q_minus = - Q
        Q_minus.backward(th.ones_like(Q_minus))

        Q_copy, Q_grad = Q_minus.to(device='cpu').clone().detach().data.numpy(), actions.grad.to(device='cpu') \
            .clone().detach().data.numpy()

        return {"value": Q_copy, "grad": Q_grad}

    def best_action(self, observation, action_init=None, nIter=5):
        bsize = observation.size()[0]

        if action_init == None:
            action_init = th.ones((bsize, self.n_actions), device=device).uniform_(0, 1)
        A = [[] for _ in range(bsize)]
        b = [[] for _ in range(bsize)]

        x = action_init
        action_min, action_max = self.actions_range
        mult_coef = action_max - action_min
        mult_coef_tensor = th.Tensor(mult_coef).to(device=device)
        action_min_tensor = th.Tensor(action_min).to(device=device)

        finished = set()
        lam = [None] * bsize

        for t in range(nIter):
            x_copy = x.cpu().clone().detach().data.numpy()
            x_copy2 = np.copy(x_copy)

            fg = self.get_gradient_batch(observation=observation, actions= action_min_tensor + x * mult_coef_tensor)

            fi, gi = fg["value"], mult_coef * fg["grad"]
            if len(finished) != bsize:

                for u in range(bsize):
                    if u not in finished:
                        Ai = gi[u]
                        bi = fi[u] - np.dot(np.transpose(gi[u]), x_copy[u])

                        A[u].append(Ai)
                        b[u].append(bi)
                        if len(A[u]) > 1:
                            lam[u] = proj_newton(np.array(A[u]), np.array(b[u]))
                            x_copy[u] = 1 / (1 + np.exp(np.array(A[u]).T.dot(lam[u])))
                            x_copy[u] = np.clip(x_copy[u], 0.001, 0.999)

                        else:
                            lam[u] = np.array([1])
                            x_copy[u] = 1 / (1 + np.exp(A[u]))
                            x_copy[u] = np.clip(x_copy[u], 0.001, 0.999)

                        A[u] = [y for i, y in enumerate(A[u]) if lam[u][i] > 0]
                        b[u] = [y for i, y in enumerate(b[u]) if lam[u][i] > 0]
                        lam[u] = lam[u][lam[u] > 0]

                        if np.linalg.norm(x_copy2[u] - x_copy[u]) < 1e-4:
                            finished.add(u)
                x = th.Tensor(x_copy).to(device)
            else:
                actions = action_min_tensor + x * mult_coef_tensor

                return {"Q": self.forward(observation, actions),
                        "actions": actions}

        actions = action_min_tensor + x * mult_coef_tensor
        return {"Q": self.forward(observation, actions),
                "actions": actions}