import numpy as np
from agents import helpers
from agents import ReinforceAgent, ReinforceWithBaselineAgent, ActorCriticAgent

class ReinforceMemoryAgent(ReinforceAgent):
    def step(self, observation, reward, done):
        '''
        This functions gets the observations and reward and outputs the next action
        that the agent should take.
        '''
        self.episode['rewards'].append(reward)
        
        # Add memory to observation
        observation = tuple(list(observation) + self.episode['actions'][-1:])
        
        action = self.start(observation, done)
        return action
    

class ReinforceWithBaselineMemoryAgent(ReinforceWithBaselineAgent):
    def step(self, observation, reward, done):
        '''
        This functions gets the observations and reward and outputs the next action
        that the agent should take.
        '''
        self.episode['rewards'].append(reward)
        
        # Add memory to observation
        observation = tuple(list(observation) + self.episode['actions'][-1:])
        
        action = self.start(observation, done)
        return action
    
    
class ActorCriticMemoryAgent(ActorCriticAgent):
    def step(self, observation, reward, done):
        '''
        This functions gets the observations and reward and outputs the next action
        that the agent should take.
        '''
        self.episode['rewards'].append(reward)
        
        # Add memory to observation
        observation = tuple(list(observation) + self.episode['actions'][-1:])
        
        action = self.start(observation, done)
        return action
