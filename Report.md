# Udacity Deep Reinforcement Learning Nanodegree
# Project 3: Collaboration & Competition

This project attempts to solve the reinforcement learning test environment "Tennis", which simulates a simplified game of table tennis.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).


## Learning Algorithm

Since this environment requires an algorithm that can handle a continuous action space, we decided to use an approach based on DDPG ([Deep Deterministic Policy Gradients](https://arxiv.org/pdf/1509.02971.pdf)), which have been shown to work well across a variety of tasks with similar complexity.

However, this is also a multi-agent environment. If we were to train DDPG individually for each agent, we would likely run into problems. The training progress of each agent's opponent effectively violates the non-stationarity assumption that most reinforcement learning algorithms require. We therefore use an adaption of DDPG, MADDPG ([Multi-Agent DDPG](https://arxiv.org/pdf/1706.02275.pdf)), designed specifically for this scenario.

Each agent maintains its own individual actor and critic model. Each agent's actor is a deterministic policy that maps from the agent's individual state observations to an individual action for that same agent. However, each agent's critic calculates action values, or Q-values, based on the global state/action space. In this way, the policy and actions of an agent's opponent are no longer a part of their environment, and the value of an agent's own state and action is assessed within in the context of the states and action choices of its peers, and not in isolation.

Q-values are calculated in a similar fashion to Q-learning (see [Deep Q-Networks](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) for details). Given an experience tuple $(s, a, r, s')$ harvested from $n$ agent's interactions with their environment, the target Q-value for agent $i$ is calculated as follows:

$$
Q_{target}(s_1,...,s_n, a_1,...,a_n) = r_i + \gamma Q(s{'}_1,...,s{'}_n, μ_i(s{'}_1),...,μ_i(s{'}_n))
$$

Where the subscript $i$ indicates the component that corresponds to agent $i$ only, $μ_i$ is the policy for agent $i$, and $\gamma$ is a discount factor. Q-values are then updated by minimising the mean-squared error between target and predicted Q-values.

Each agent individually measures the performance of its policy as follows:

$$
J_{μ_i} = Q(s_1,...,s_i,...,s_n, a_1,...,μ_i(s_i),...,a_n)
$$

The policy is improved by maximising this term.

We use neural networks to represent both actor and critic and train them using Adam optimisation. We also use two additional ideas from the [Deep Q-Networks](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) paper and also recommended in the [MADDPG](https://arxiv.org/pdf/1706.02275.pdf) paper. Firstly, we use a replay buffer to collect samples and then training on random batches from this replay buffer. Secondly, we maintain separate target actor and critic networks for calculating target values. We then soft-update these target networks over time, which leads to more stable learning.


# Neural network architecture

Both networks are distinct, yet share essentially the same architecture but with different outputs.

Below is the summary for the actor/policy network. On the hidden layers we use the `ReLU` activation function. On the outputs we use `tanh` to produce actions that are in the desired range -1 to 1.

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                   [-1, 32]           1,088
                Linear-2                   [-1, 32]           1,056
                Linear-3                    [-1, 4]             132
    ================================================================
    Total params: 2,276
    Trainable params: 2,276
    Non-trainable params: 0
    ----------------------------------------------------------------

Below is the summary for the critic/value network. Again we use `ReLU` on the hidden layers. The final layer has no activation function, since Q-values can, in theory, take on any value.

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Linear-1                   [-1, 32]           1,088
                Linear-2                   [-1, 32]           1,056
                Linear-3                    [-1, 1]              33
    ================================================================
    Total params: 2,177
    Trainable params: 2,177
    Non-trainable params: 0
    ----------------------------------------------------------------


## Plot of Rewards

As required, the agent is able to receive an average maximum reward (over 100 episodes) of at least 0.5. This is reached after just 173 episodes.

![Reward Graph of PPO](https://github.com/chris838/reacher/blob/master/ppo-reacher.png)


## Ideas for Future Work

The most difficult aspect of this project was finding the right hyperparameters that would provide good performance. Initially, we started with the parameters provided in the original MADDPG paper, but the proved to perform poorly. By reducing the learning rate and increasing the training frequency, as well as training for more episodes, we eventually found a working solution.

Given how long the algorithm takes to converge and how sensitive it is to certain parameters, it seems highly likely that significant improvement could be made by exploring different parameter settings. This search proved time consuming, however, given that even our most successful training attempts showed no improvement until after several thousand episodes of seemingly degenerate behaviour.

Therefore, it would seem prudent to find a method of assessing training progress that goes beyond just looking at cumulative reward and actor/critic loss, so that we can try and improve training efficiency without spending hours waiting for results. Installing and configuring a utility like 'tensorboardX' would likely be the easiest way of doing this.

Beyond tweaking parameters, we might also consider using prioritised replay. Clearly, many of the episodes our agent experiences contain little information, in particular when the agent fails to hit the ball at all. In contrast, some episodes are highly information - a back-and-forth rally can occasionally be seen when watching random agents play.


# Hyperparameters

    | Hyper-parameter Name 	  | Value 	|
    |-------------------------|---------|
    | Adam learning rate  	  | 3e-4  	|
    | Discount (gamma)    	  | 0.99  	|
    | Soft-update (tau)       | 0.01    |
    | Batch size              | 512     |
    | Max. episode length     | 2000    |
    | Replay buffer min size  | 20480   |
    | Replay buffer max size  | 1000000 |
    | Train every # steps     | 4       |
    | Noise start level       | 1.0     |
    | Noise decay rate        | 0.9999  |
