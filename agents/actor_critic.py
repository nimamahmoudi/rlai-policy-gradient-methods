import numpy as np
from agents import helpers

class ActorCriticAgent:
    def __init__(self, alpha_theta=0.01, alpha_w=0.01, precision_class=np.float16):
        '''
        Initialize the REINFORCE agent. alpha is the learning rate for gradient
        descent algorithm. precision_class is the numpy class of variable that
        we use for our calculations.
        '''
        self.episode = {}
        # Number of actions that we can take. This algorithm will take one of 0,1,...,N-1 as action.
        self.num_of_actions = 4
        self.discount_factor = 1
        self.alpha_theta = alpha_theta
        self.alpha_w = alpha_w
        self.theta_default_val = 0
        self.v_hat_default_val = 1
        # The class of numpy that we use for our calculations.
        self.precision_class = precision_class
        self.theta = {}
        self.v_hat = {}

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
        
    def get_v_hat_val(self, idx):
        if idx in self.v_hat:
            return self.v_hat[idx]
        else:
            self.v_hat[idx] = self.v_hat_default_val
            return self.v_hat_default_val

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
        if len(self.episode['rewards']) > 0:
            obs = self.episode['observations'][-1]
            action = self.episode['actions'][-1]

            R = self.episode['rewards'][-1]
            idx = helpers.convert_s_a(obs, action)

            v_hat = self.get_v_hat_val(obs)
            v_hat_p = self.get_v_hat_val(observation)  # next state value
            pi_a = self.get_action_vals_for_obs(obs)[action]
            old_theta = self.get_theta_val(idx)

            delta = R + (self.discount_factor * v_hat_p) - v_hat
            v_hat_new = v_hat + self.alpha_w * delta
            new_theta = old_theta + self.alpha_theta * (
                    delta * (1 - pi_a) * 1
            )

            self.v_hat[obs] = v_hat_new
            self.set_theta_val(idx, new_theta)
        
        # Take action
        action_values = self.get_action_vals_for_obs(observation)
        # pick a random action based on the probs
        action = np.random.choice(list(range(self.num_of_actions)), p=action_values)

        self.episode['observations'].append(observation)
        self.episode['actions'].append(action)
        
        # Make update for last step
        if done:
            obs = self.episode['observations'][-1]
            action = self.episode['actions'][-1]
            R = self.episode['rewards'][-1]
            idx = helpers.convert_s_a(obs, action)
            v_hat = self.get_v_hat_val(obs)
            v_hat_p = 0
            pi_a = self.get_action_vals_for_obs(obs)[action]
            old_theta = self.get_theta_val(idx)
            delta = R + (self.discount_factor * v_hat_p) - v_hat
            v_hat_new = v_hat + self.alpha_w * delta
            new_theta = old_theta + self.alpha_theta * (
                    delta * (1 - pi_a) * 1
            )
            self.v_hat[obs] = v_hat_new
            self.set_theta_val(idx, new_theta)

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
        self.reset_episode()

    def reset_episode(self):
        self.episode = {
            'actions': [],
            'observations': [],
            'rewards': [],
        }
