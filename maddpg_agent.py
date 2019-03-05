import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorNetwork(nn.Module):

    def __init__(self, in_size, h1_size, h2_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # x is a 2D vector (a force that is applied to the agent)
        # We bound the norm of the vector to be between 0 and 10
        norm = torch.norm(x)
        if norm > 0:
            return 10.0 * (f.tanh(norm)) * (x / norm)
        else:
            10 * x

class CriticNetwork(nn.Module):

    def __init__(self, in_size, h1_size, h2_size, out_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_size, h1_size)
        self.fc2 = nn.Linear(h1_size, h2_size)
        self.fc3 = nn.Linear(h2_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Critic returns the q-value
        return x


class MaddpgAgent():

    def __init__(self, i, state_space_size, action_space_size):

        self.i = i
        self.actor  = ActorNetwork(state_space_size, 16, 8, 2)
        self.critic = CriticNetwork(state_space_size + action_space_size, 16, 8, 1)

        # Initialize a random process N for action exploration
        # TODO

    def act(self, state, noise_level=0):

        # Select action:
        #    a = µ_θ (o) + N_t
        # w.r.t. the current policy and exploration noise
        # TODO

        return np.random.random( self.action_space_size )

    def tensorise_sample(self, sample):
        (s_s, a_s, r_s, s_prime_s) = zip(*tuple(sample))
        s_s = torch.tensor(s_s).float()
        a_s = torch.tensor(a_s).float()
        r_s = torch.tensor(r_s).float()
        s_prime_s = torch.tensor(s_prime_s).float()
        return (s_s, a_s, r_s, s_prime_s)

    def update(self, sample, next_actions):

        # Unzip the list of sample tuples (s,a,r,s_prime) into seperate
        # tensors of each component.
        (s_s, a_s, r_s, s_prime_s) = self.tensorise_sample(sample)
        a_prime_s = torch.tensor(next_actions).float()

        # For each sample, calculate the Q-value target y for this agent
        # specifically:
        #   y = r_i + γ * Q^{µ'}_i(x', a'_1, . . . , a'_N ) | a'_k = µ'_k(o_k)


        pass
