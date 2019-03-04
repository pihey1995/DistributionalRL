<h1>Distributional Reinforcement Learning</h1>
This repository is by Paul-Ambroise D. and Pierre-Alexandre K. and contains the PyTorch source code to reproduce the 
results of Bellemare and al. ["A Distributional Perspective on Reinforcement Learning"](https://arxiv.org/abs/1707.06887).

<h2>Requirements</h2>
- Python 3.6
- Torch
- OpenAI gym

<h2>Results</h2>
We used the categorical algorithm to solve [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/).

The following results were not optimized over different hyperparameters, so there is room for improvement.

![](/results/figs/test_score.png)

The evolution of the distribution for the [0, 0, 0, 0] state is the following:
![](/results/figs/gifs/seed-1.gif)

<h2>Discussion</h2>
We want to extend the work of Bellemare and al. to continuous action using either ICNN, CEM or NAF to handle continuous actions.
An ICNN implementation is yet available but needs optimization.

Implicit : étendre aux actions continues
https://arxiv.org/pdf/1806.06923.pdf
QUOTA : https://arxiv.org/pdf/1811.02073.pdf
Quantile regression : c51 qrdqn
DISTRIBUTED DISTRIBUTIONAL DETERMINISTIC POLICY GRADIENTS: https://openreview.net/pdf?id=SyZipzbCb

