import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, in_size, h1_size, h2_size, out_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(in_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x is a 2D vector (a force that is applied to the agent)
        # We bound the norm of the vector to be between 0 and 1
        norm = torch.norm(x)
        if norm > 0: return (x / norm)
        else:        return  x

class CriticNetwork(nn.Module):

    def __init__(self, num_agents, state_space_size, action_space_size,
                 h1_size, h2_size, out_size):
        super(CriticNetwork, self).__init__()

        in_size = num_agents*(state_space_size + action_space_size)
        self.fc1 = nn.Linear(in_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, out_size)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Critic returns the q-value
        return x


class MaddpgAgent():

    def __init__(self, i, num_agents, state_space_size, action_space_size):

        self.i = i
        self.num_agents = num_agents
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size

        self.discount = 0.95
        self.tau = 0.01
        self.lr = 0.01

        # Create the ANN models. We use target networks to make predictions
        # and apply updates to local networks, which are then soft-updated
        # from the local networks over time. This helps stabilise learning.
        self.actor  = ActorNetwork(
            state_space_size, 16, 8, 2)
        self.actor_target  = ActorNetwork(
            state_space_size, 16, 8, 2)
        self.critic = CriticNetwork(
            num_agents, state_space_size, action_space_size, 16, 8, 1)
        self.critic_target = CriticNetwork(
            num_agents, state_space_size, action_space_size, 16, 8, 1)

        self.actor_optimiser    = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimiser   = Adam(self.critic.parameters(), lr=self.lr)

    def act(self, state, epsilon=0, target_actor=False):

        # Select action:
        #    a = µ_θ (o) + N_t
        # w.r.t. the current policy and exploration noise
        with torch.no_grad():
            if target_actor:
                action = self.actor_target(torch.from_numpy(state).float())
            else:
                action = self.actor(torch.from_numpy(state).float())

        # TODO - add noise

        return action.detach().numpy()

    def tensorise_sample(self, sample):
        (s, a, r, s_prime) = zip(*tuple(sample))
        s       = torch.tensor(s).float()
        a       = torch.tensor(a).float()
        r       = torch.tensor(r).float()
        s_prime = torch.tensor(s_prime).float()
        return (s, a, r, s_prime)

    def update(self, sample, next_actions):

        # Unzip the list of sample tuples (s,a,r,s_prime) into seperate
        # tensors of each component.
        (s, a, r, s_prime) = self.tensorise_sample(sample)
        a_prime = torch.tensor(next_actions).float()

        # Update the local actor/critic networks
        self.update_critic(s, a, r, s_prime, a_prime)
        self.update_actor(s, a, r, s_prime, a_prime)

        # Soft update the target actor/critic networks
        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)


    def update_critic(self, s, a, r, s_prime, a_prime):

        self.critic.zero_grad()

        # For each sample, calculate the Q-value target, y, for this agent i
        # specifically:
        #   y = r_i + γ * Q_i(x'_1, ... , x'_N, a'_1, ... , a'_N) | a'_k = µ'_k(o_k)
        next_states = torch.flatten(s_prime, start_dim=1)
        next_actions = torch.flatten(a_prime, start_dim=1)
        y = r + self.discount * self.critic_target(next_states, next_actions)

        # Calculate the critic loss for this agent i. For each sample tuple,
        # this is given by:
        #   L(θ_i) = (y − Q_i(s_1, ... , s_N, a_1, ... , a_N))^2
        # We then average this squared error term over all the sample tuples
        # (i.e. we calculate the MSE loss over the sample)
        states = torch.flatten(s, start_dim=1).requires_grad_(True)
        actions = torch.flatten(a, start_dim=1).requires_grad_(True)
        critic_loss = F.mse_loss(y.detach(), self.critic(states, actions))

        # Optimise
        critic_loss.backward()
        self.critic_optimiser.step()


    def update_actor(self, s, a, r, s_prime, a_prime):

        self.actor.zero_grad()
        self.critic.zero_grad()

        # Calculate the policy loss for this agent i. For each sample tuple,
        # this is given by:
        #   L(w_i) = -1 * Q_i(s_1, .. s_i .. , s_N, a_1, .. μ(s_i) .. , a_N)
        # Where μ(s) is the (deterministic) policy determined actions. The
        # gradient of s_i influences the gradients of the parameters w_i,
        # so it's important we track this. The loss is then averaged over all
        # the sample tuples

        # First, apply the policy to determine the action(s) for all tuples in
        # the sample, for this agent i only, making sure we track the gradients.
        this_agent_states = s[:, self.i].requires_grad_(True)
        this_agent_actions = self.actor(this_agent_states)

        # We already have the actions for the other agents. Splice them together
        # Again, this is for all tuples in the sample
        before = a[:, :self.i] # actions for agents j, where j < i
        after = a[:, self.i + 1:] # actions for agents j, where j > i
        a_modified = torch.cat([before.detach(),
                                this_agent_actions.unsqueeze(1),
                                after.detach()], dim=1)

        # Now we can define the policy (actor) loss, at this point we average
        # over all sample tuples in our sample. We also multiple by -1 since
        # otherwise we have defined the actor's performance, instead of loss.
        states = torch.flatten(s, start_dim=1).requires_grad_(True)
        actions = torch.flatten(a_modified, start_dim=1).requires_grad_(True)
        actor_loss = -1 * self.critic(states, actions).mean()

        # Optimise
        actor_loss.backward()
        self.actor_optimiser.step()


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )
