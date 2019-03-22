import numpy as np
from agents import helpers

class ReinforceAgent:
    def __init__(self, alpha=0.001, precision_class=np.float16):
        '''
        Initialize the REINFORCE agent. alpha is the learning rate for gradient
        descent algorithm. precision_class is the numpy class of variable that
        we use for our calculations.
        '''
        self.episode = {}
        # Number of actions that we can take. This algorithm will take one of 0,1,...,N-1 as action.
        self.num_of_actions = 4
        self.discount_factor = 1
        self.alpha = alpha
        self.theta_default_val = 0
        # The class of numpy that we use for our calculations.
        self.precision_class = precision_class
        self.theta = {}

        # Reset variables that keep episode information
        self.reset_episode()

    def get_theta_val(self, idx):
        '''
        Fetches the theta value from the corresponding dictionary. Returns and sets the default value
        for indices that we havn't seen before.
        '''
        if idx in self.theta:
            return self.theta[idx]
        else:
            self.theta[idx] = self.theta_default_val
            return self.theta_default_val

    def get_pref(self, idx):
        '''
        Get the preference (shown in book as h) from theta.
        '''
        # h = theta[s]
        theta = self.get_theta_val(idx)
        return theta

    def get_action_vals(self, all_idx):
        '''
        Get the probability of taking actions for current observation. It calculates them
        for all possible actions in current observation.
        '''
        all_pref = []
        for idx in all_idx:
            all_pref.append(self.get_pref(idx))
        all_pref = np.array(all_pref).astype(self.precision_class)

        # normalize theta
        theta_mean = np.mean(all_pref)
        # theta_mean = 0
        for idx in all_idx:
            val = self.get_theta_val(idx) - theta_mean
            # Just so that we won't overflow the system
            if val < -10000:
                val = -10000
            if val > 10000:
                val = 10000
            self.set_theta_val(idx, val)

        # fix overflow issue, since underflow is much better (zeros vs inf)
        all_pref = all_pref - np.max(all_pref)

        # calculate softmax
        all_pi = np.exp(all_pref)
        all_pi = all_pi / np.sum(all_pi)

        return all_pi

    def set_theta_val(self, idx, val):
        '''
        Set the theta for an idx in the dict.
        '''
        self.theta[idx] = val

    def get_action_vals_for_obs(self, obs):
        '''
        Get probability of taking actions for all possible actions of an observation.
        '''
        all_idx = [helpers.convert_s_a(obs, i) for i in range(self.num_of_actions)]
        action_values = self.get_action_vals(all_idx)
        return action_values

    def start(self, observation, done=False):
        '''
        Taking an action in the start. This function is also used
        for next steps, but we add a pre-step there.
        '''
        self.episode['observations'].append(observation)

        action_values = self.get_action_vals_for_obs(observation)
        # pick a random action based on the probs
        action = np.random.choice(list(range(self.num_of_actions)), p=action_values)

        self.episode['actions'].append(action)

        return action

    def step(self, observation, reward, done):
        '''
        This functions gets the observations and reward and outputs the next action
        that the agent should take.
        '''
        self.episode['rewards'].append(reward)
        action = self.start(observation, done)
        return action

    def update_for_episode(self):
        '''
        Updates that we need to do when the episode is finished.
        '''
        actions = self.episode['actions']
        observations = self.episode['observations']
        rewards = self.episode['rewards']

        # Monte Carlo Specific
        g = 0
        times = np.arange(0, len(rewards))
        for t in times[::-1]:
            g = self.discount_factor * g + rewards[t]  # t'th value in rewards array is R_(t+1) in the book
            obs = observations[t]
            action = actions[t]
            # Get the index that corresponds to obs and action
            idx = helpers.convert_s_a(obs, action)

            # Get the probability of taking different actions
            pi_a = self.get_action_vals_for_obs(obs)[action]

            # Calculate the new theta based on the old theta and our update policy.
            old_theta = self.get_theta_val(idx)
            new_theta = old_theta + self.alpha * (
                    g * (1 - pi_a) * 1
            )

            # Set the new theta value
            self.set_theta_val(idx, new_theta)

        self.reset_episode()

    def reset_episode(self):
        self.episode = {
            'actions': [],
            'observations': [],
            'rewards': [],
        }
