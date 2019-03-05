import torch

class MaddpgAgent():

    def __init__(self):

        # Initialize a random process N for action exploration
        pass

    def act(self, state, noise_level=0):

        # Select action:
        #    a = µ_θ (o) + N_t
        # w.r.t. the current policy and exploration noise
        pass

    def tensorise_sample(self, sample):
        ss, as, rs, s_primes = zip(tuple(sample))
        ss = torch.tensor(ss).float()
        as = torch.tensor(as).float()
        rs = torch.tensor(rs).float()
        s_primes = torch.tensor(s_primes).float()
        return (ss, as, rs, s_primes)

    def update(self, sample, next_actions):

        # Unzip the list of sample tuples (s,a,r,s_prime) into seperate
        # tensors of each component.
        (ss, as, rs, s_primes) = self.tensorise_sample(sample)
        a_primes = torch.tensor(next_actions).float()

        # For each sample, calculate the Q-value target y for this agent
        # specifically:
        #   y = r_i + γ * Q^{µ'}_i(x', a'_1, . . . , a'_N ) | a'_k = µ'_k(o_k)


        pass
