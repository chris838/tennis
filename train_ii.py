import pdb
import numpy as np

from mp_envs import make_env
from replay_buffer import ReplayBuffer
from maddpg_agent import MaddpgAgent

# %%

# Multi-Agent Deep Deterministic Policy Gradient for N agents

# Config
num_episodes = 5000
batch_size = 1024  # how many episodes to process at once
max_episode_length = 25
environment_name = "simple_adversary"
replay_buffer_size_max = int(1e6)
train_every_steps = 100  # steps between training updates

# Exploration factor, this slowly decreases to 0
epsilon = 1.0
epsilon_decay = 0.9999

# Load the environment
env = make_env(environment_name)
num_agents = env.n
print(f"Environment has {num_agents} agents")

# Create the agents
agents = []
global_state_space_size = sum([o.shape[0] for o in env.observation_space])
global_action_space_size = sum([a.shape[0] for a in env.action_space])
for i in range(num_agents):
    state_space_size = env.observation_space[i].shape[0]
    action_space_size = env.action_space[i].shape[0]
    print(f"Agent {i}: state space: {state_space_size}; \
            action space {action_space_size}.")
    agents.append(MaddpgAgent(
        i, num_agents, state_space_size, action_space_size,
        global_state_space_size, global_action_space_size))

# Don't start learning until we have more episodes recorded than we need
# samples to fill our batch (i.e. we're only taking on average 1-2 samples from
# each episode).
min_samples_required = batch_size * max_episode_length

# Create the replay buffer
replay_buffer = ReplayBuffer(
    max_size=replay_buffer_size_max, min_samples_required=min_samples_required)

# Track progress
episode_rewards = []

# Iterate over episodes
train_step = 0
for episode in range(1, num_episodes):

    # Receive initial state vector s
    #   s = (s_1, . . . , s_N)
    s = env.reset()


    episode_rewards.append( np.array( [0] * num_agents) )
    for t in range(1, max_episode_length):

        # For each agent i, select actions:
        #   a = (a_1, . . . , a_N)
        # using the current policy and exploration noise, which we decay
        a = [agent.act(state, epsilon=epsilon)
             for agent, state in zip(agents, s)]
        epsilon *= epsilon_decay

        # Execute actions a = (a_1, . . . , a_N)
        # Observe:
        #   Reward r = (r_1, . . . , r_N)
        #   Next-state vector s' = (s'_1, . . . , s'_N)
        s_prime, r, *_ = env.step(a)

        # Store (s, a, r, s') in replay buffer D
        replay_buffer.append((s, a, r, s_prime))

        # Record progress
        episode_rewards[-1] = episode_rewards[-1] + r

        # Advance
        s = s_prime
        train_step += 1

        # Periodically (after a certain number of steps) run update/training
        if train_step % train_every_steps == 0:
            if replay_buffer.has_enough_samples():

                # Sample replay buffer
                sample = replay_buffer.sample(batch_size=batch_size)

                # For every sample tuple, each agent needs to know which action
                # would be chosen under the policy of the other agents in the
                # next state s', in order to calculate q-values.
                next_actions = [[
                     agent.act(next_state, target_actor=True)
                     for agent, next_state in zip(agents, s_prime)]
                    for (s, a, r, s_prime) in sample]

                # Update/train all the agents
                for agent in agents:
                    agent.update(sample, next_actions)

    if episode % 100 == 0:
        print(f"Average episode return over last 100 episodes: \
        {np.array(episode_rewards[-100:]).mean(axis=0)}")
