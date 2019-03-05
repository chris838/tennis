from mpe_envs import make_env

# Multi-Agent Deep Deterministic Policy Gradient for N agents

num_episodes = 100

# Load the environment
env = make_env("simple")

# Detect the number of agents (N)
num_agents = env.n

# Amplitude of OU noise, this slowly decreases to 0
noise_level = 2
noise_reduction = 0.9999

# Create the agents
agents = []
for i in range(num_agents):
    agents.append(MaddpgAgent(i,
                           state_space=env.observation_space,
                           action_space=env.action_space))

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
        noise_level *= noise_reduction

        # Execute actions a = (a_1, . . . , a_N)
        # Observe reward r = (r_1, . . . , r_N) and next state vector s_prime
        s_prime, r, *_ = env.step(a)

        # Store (s, a, r, s_prime) in replay buffer D
        pass

        # Advance
        s = s_prime

        for agent in agents:

            # Sample replay buffer
            pass

            # Update agent actor/critic with replay buffer sample
            pass
