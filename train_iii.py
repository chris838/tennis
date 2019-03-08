from unityagents import UnityEnvironment

import pdb, pickle

import numpy as np
import holoviews as hv
import pandas as pd
import xarray as xr

from replay_buffer import ReplayBuffer
from maddpg_agent import MaddpgAgent

env = UnityEnvironment(file_name="./Tennis.app")

# Get the default brain and reset env
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]

# Number of agents
num_agents = len(env_info.agents)
print(f"Number of agents: {num_agents}")

# Size of the global state/action space (across all agents)
actions = env_info.previous_vector_actions
states = env_info.vector_observations
global_state_space_size = states.flatten().shape[0]
global_action_space_size = actions.flatten().shape[0]
print(f"Global states: {global_state_space_size}")
print(f"Global actions: {global_action_space_size}")

# Size of the local state/action space (for each agent individually)
action_space_size = brain.vector_action_space_size
state_space_size = brain.num_stacked_vector_observations * brain.vector_observation_space_size
print(f"Local states: {state_space_size}")
print(f"Local actions: {action_space_size}")

# Examine the state space
print('The state for the first agent looks like:', states[0])


def train(
    num_episodes = 500,
    batch_size = 1024,
    max_episode_length = 250,
    expected_episode_length = 5,
    replay_buffer_size_max = int(1e6),
    train_every_steps = 100,
    noise_level = 1.0,
    noise_decay = 0.9999,
    print_episodes = 20
):

    print(f"------------------------------------------------")
    print(f"New training run.")
    print(f"    num_episodes: {num_episodes}")
    print(f"    batch_size: {batch_size}")
    print(f"    max_episode_length: {max_episode_length}")
    print(f"    expected_episode_length: {expected_episode_length}")
    print(f"    replay_buffer_size_max: {replay_buffer_size_max}")
    print(f"    train_every_steps: {train_every_steps}")
    print(f"    noise_level: {noise_level}")
    print(f"    noise_decay: {noise_decay}")

    # Create the agents
    agents = []
    for i in range(num_agents):
        print(f"Agent {i}: state space: {state_space_size}; \
                action space {action_space_size}.")
        agents.append(MaddpgAgent(
            i, num_agents, state_space_size, action_space_size,
            global_state_space_size, global_action_space_size))

    # Don't start learning until we have more episodes recorded than we need
    # samples to fill our batch (i.e. we're only taking on average 1-2 samples from
    # each episode).
    min_samples_required = batch_size * expected_episode_length

    # Create the replay buffer
    replay_buffer = ReplayBuffer(
        max_size=replay_buffer_size_max, min_samples_required=min_samples_required)

    # Track progress
    episode_rewards = []
    loss = []

    # Iterate over episodes
    train_step = 0
    is_learning = False
    for episode in range(1, num_episodes):

        # Receive initial state vector s
        #   s = (s_1, . . . , s_N)
        env_info = env.reset(train_mode=True)[brain_name]
        s = env_info.vector_observations

        episode_rewards.append( np.array( [0] * num_agents) )
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
            env_info= env.step(a)[brain_name]
            s_prime = env_info.vector_observations
            r = env_info.rewards
            dones = env_info.local_done

            # Store (s, a, r, s', dones) in replay buffer D
            replay_buffer.append((s, a, r, s_prime, dones))

            # Record progress
            episode_rewards[-1] = episode_rewards[-1] + r

            # Advance
            s = s_prime
            train_step += 1

            # Periodically (after a certain number of steps) run update/training
            if train_step % train_every_steps == 0:
                if replay_buffer.has_enough_samples():

                    if not is_learning:
                        print(f"Started learning at time {train_step}")
                        is_learning = True

                    # Sample replay buffer
                    sample = replay_buffer.sample(batch_size=batch_size)

                    # For every sample tuple, each agent needs to know which action
                    # would be chosen under the policy of the other agents in the
                    # next state s', in order to calculate q-values.
                    next_actions = [[
                         agent.act(next_state, target_actor=True)
                         for agent, next_state in zip(agents, s_prime)]
                        for (s, a, r, s_prime, dones) in sample]

                    # Update/train all the agents
                    per_agent_loss = []
                    for agent in agents:
                        actor_loss, critic_loss = agent.update(sample, next_actions)
                        per_agent_loss.append((actor_loss, critic_loss))
                    loss.append(per_agent_loss)

                    #print(f"loss: {np.array(loss[-10:]).mean(axis=(0,1))}")

            # Terminate episode early if done
            if any(dones):
                break

        if episode % print_episodes == 0:
            print(f"t: {train_step}, e: {episode}, noise: {noise_level:.2f}. " + \
                  f"Average last {print_episodes} episode return: " + \
                  f"{np.array(episode_rewards[-print_episodes:]).mean(axis=0)}")

    print(f"Final average reward over entire run: {np.mean(episode_rewards)}")
    return np.array(episode_rewards).mean(axis=1), np.array(loss).mean(axis=1), agents


results, loss, agents = train(
    num_episodes=200,
    batch_size=1024,
    train_every_steps=100,
    max_episode_length = 250,
    expected_episode_length = 1,
    noise_level = 1.0,
    noise_decay = 0.9999,
    print_episodes = 20
)
