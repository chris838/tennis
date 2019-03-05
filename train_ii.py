from mpe_envs import make_env
from replay_buffer import ReplayBuffer
from tensor_utils import tensorise_sample

# Multi-Agent Deep Deterministic Policy Gradient for N agents

# Config
num_episodes = 100
batch_size = 1024  # how many episodes to process at once
max_episode_length = 50
environment_name = "simple"
replay_buffer_size_max = 1e6
train_every_steps = 100  # steps between training updates

# Amplitude of OU noise, this slowly decreases to 0
noise_level = 2
noise_decay = 0.9999

# Load the environment
env = make_env(environment_name)

# Detect the number of agents (N)
num_agents = env.n

# Create the replay buffer
min_samples_required = batch_size * max_episode_length
replay_buffer = ReplayBuffer(
    max_size=replay_buffer_size, min_samples_required=min_samples_required)

# Create the agents
agents = []
for i in range(num_agents):
    agents.append(MaddpgAgent(i,
                              state_space=env.observation_space,
                              action_space=env.action_space))

# Iterate over episodes
train_step = 0
for episode in range(1, num_episodes):

    # Receive initial state vector s
    #   s = (s_1, . . . , s_N)
    x = env.reset()

    for t in range(1, max_episode_length):

        # For each agent i, select actions:
        #   a = (a_1, . . . , a_N)
        # using the current policy and exploration noise, which we decay
        a = [agent.act(state, noise_level=noise_level)
             for agent, state in zip(agents, s)]
        noise_level *= noise_decay

        # Execute actions a = (a_1, . . . , a_N)
        # Observe:
        #   Reward r = (r_1, . . . , r_N)
        #   Next-state vector s' = (s'_1, . . . , s'_N)
        s_prime, r, *_ = env.step(a)

        # Store (s, a, r, s') in replay buffer D
        replay_buffer.append((s, a, r, s_prime))

        # Advance
        s = s_prime

        # Periodically (after a certain number of steps) run update/training
        if train_step % train_every_steps == 0:
            if replay_buffer.has_enough_samples():

                # Sample replay buffer
                sample = replay_buffer.sample(batch_size=)

                # For every sample tuple, each agent needs to know which action
                # would be chosen under the policy of the other agents in the
                # next state s', in order to calculate q-values.
                next_actions = [[
                     agent.act(next_state)
                     for agent, next_state in zip(agents, s_prime)]
                    for (s, a, r, s_prime) in sample]

                # Update/train all the agents
                for agent in agents:
                    agent.update(sample, next_actions)
