###THIS CODE HAS BEEN MOSTLY RETRIEVED ON THE GITHUB OF NITARSHAN"
#https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb

import torch
from torch import nn
import torch.nn.functional as F
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    def sigma(self):
        return torch.log1p(torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum()


class Bayesian_layer(nn.Module):
    def __init__(self, input, output,PI, SIGMA_1, SIGMA_2):
        super(Bayesian_layer, self).__init__()
        self.weight_mu = nn.Parameter(torch.Tensor(output, input).uniform_(-1,1))
        self.weight_rho = nn.Parameter(torch.Tensor(output, input).uniform_(-1,1))
        self.weights =  Gaussian(self.weight_mu, self.weight_rho)
        self.bias_mu = nn.Parameter(torch.Tensor(output).uniform_(-1,1))
        self.bias_rho = nn.Parameter(torch.Tensor(output).uniform_(-1,1))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)

        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)

        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu

        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)


class Bayesian_QNetwork(nn.Module):
    def __init__(self, input, output, PI, SIGMA_1, SIGMA_2,  hiddens=[64], seed=0):
        super(Bayesian_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.hiddens = hiddens
        if len(self.hiddens) == 0:
            self.fc0 = nn.Linear(input, output)
        for i in range(len(hiddens)):
            if i == 0:
                setattr(self, "fc{}".format(i), Bayesian_layer(input, self.hiddens[i], PI, SIGMA_1, SIGMA_2))
            else:
                setattr(self, "fc{}".format(i), Bayesian_layer(self.hiddens[i-1], self.hiddens[i], PI, SIGMA_1, SIGMA_2))

        setattr(self, "fc{}".format(len(hiddens)), Bayesian_layer(self.hiddens[-1], output, PI, SIGMA_1, SIGMA_2))

    def forward(self, x, sample):
        if len(self.hiddens) == 0:
            return self.fc0(x, sample)
        else:
            for i in range(0, len(self.hiddens)):
                x = F.relu(getattr(self, "fc{}".format(i))(x, sample))
            return F.log_softmax(getattr(self, "fc{}".format(len(self.hiddens)))(x, sample))

    def log_prior(self):
        log_prior = 0
        for i in range(0, len(self.hiddens)+1):
            log_prior+= getattr(self, "fc{}".format(i)).log_prior
        return log_prior

    def log_variational_posterior(self):
        log_variational_posterior = 0
        for i in range(0, len(self.hiddens)+1):
            log_variational_posterior+= getattr(self, "fc{}".format(i)).log_variational_posterior
        return log_variational_posterior

    def sample_elbo(self, input, target, samples=SAMPLES):
        outputs = torch.zeros(samples, BATCH_SIZE, CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        for i in range(samples):
            outputs[i] = self(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        negative_log_likelihood = F.nll_loss(outputs.mean(0), target, size_average=False)
        loss = (log_variational_posterior - log_prior) / NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood