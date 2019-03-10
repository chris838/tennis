# Solving Tennis using Multi-Agent Deep Deterministic Policy Gradients

This project attempts to solve the reinforcement learning test environment "Tennis", which simulates a simplified game of table tennis, using MADDPG (Multi-Agent Deep Deterministic Policy Gradients) [Lowe et al](https://arxiv.org/abs/1706.02275).


## Description

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


## Installation

### Install deep reinforcement learning repository

1. Clone [deep reinforcement learning repository](https://github.com/udacity/deep-reinforcement-learning)
2. Follow the instructions to install necessary [dependencies](https://github.com/udacity/deep-reinforcement-learning#dependencies)


### Download the Unity Environment

1. Download environment for your system into this repository root

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
* Headless: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)

2. Unzip (or decompress) the archive


### Run the project

1. Start the jupyter server by running `jupyter notebook`
2. Open the `MADDPG_Tennis_Multi_Agent.ipynb` notebook
3. Change the kernel to `drlnd`
4. You should be able to run all the cells


## Weights

The directory `saved_models` contains saved weights and training state for the two agents. Follow the instructions in the notebook to load them.
