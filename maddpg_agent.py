import torch, pdb
import numpy as np

class MaddpgAgent():

    def __init__(self, i, state_space, action_space):

        self.i = i
        self.state_space = state_space
        self.action_space = action_space

        # Initialize a random process N for action exploration
        # TODO

    def act(self, state, noise_level=0):

        # Select action:
        #    a = µ_θ (o) + N_t
        # w.r.t. the current policy and exploration noise
        # TODO

        return np.random.random( self.action_space.n )

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
