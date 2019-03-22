import random

__version__ = '0.1'


class SmartVac:
    MOVE_UP = 0
    MOVE_RIGHT = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3

    def __init__(self, terminal_rewards=(+1,-1), step_reward=-0.1):
        self.y = 0
        self.x = 0
        self.num_cols = 5
        self.reset()
        self.terminal_rewards = terminal_rewards
        self.step_reward = step_reward


    def reset(self):
        self.y = 1
        self.x = random.choice(list(range(self.num_cols)))

        obs = self.get_obs()

        return obs

    def step(self, action):
        actions = self.get_actions()
        done = False
        reward = self.step_reward

        # Terminal state, get the final reward
        if self.y == 0:
            done = True
            if self.x == 2:
                reward = self.terminal_rewards[0]
            else:
                reward = self.terminal_rewards[1]
        else:
            # if action is possible and can be done, else do nothing
            if action in actions:
                if action == SmartVac.MOVE_RIGHT:
                    self.x += 1
                elif action == SmartVac.MOVE_LEFT:
                    self.x -= 1
                elif action == SmartVac.MOVE_DOWN:
                    print("Something went wrong!!!!")
                elif action == SmartVac.MOVE_UP:
                    self.y -= 1

        return self.get_obs(), reward, done

    def get_obs(self):
        sensors = [0] * 4

        if self.y == 0:
            sensors[SmartVac.MOVE_UP] = 1
            sensors[SmartVac.MOVE_LEFT] = 1
            sensors[SmartVac.MOVE_RIGHT] = 1

        if self.y == 1:
            sensors[SmartVac.MOVE_DOWN] = 1

            if self.x == 1 or self.x == 3:
                sensors[SmartVac.MOVE_UP] = 1

        if self.x == 0:
            sensors[SmartVac.MOVE_LEFT] = 1

        if self.x == (self.num_cols - 1):
            sensors[SmartVac.MOVE_RIGHT] = 1

        return tuple(sensors)

    def get_actions(self):
        actions = []
        obss = self.get_obs()

        # If there is not a wall there, we can go that way
        for i, obs in enumerate(obss):
            if obs == 0:
                actions.append(i)

        return actions

    def __str__(self):
        obs = self.get_obs()
        actions = self.get_actions()
        ret = "Observation: "
        ret += "\r\n"

        ret += "\t" + str(obs[SmartVac.MOVE_UP])
        ret += "\r\n"
        ret += str(obs[SmartVac.MOVE_LEFT]) + "\t\t" + str(obs[SmartVac.MOVE_RIGHT])
        ret += "\r\n"
        ret += "\t" + str(obs[SmartVac.MOVE_DOWN])

        ret += "\r\n"

        return ret


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    episode_count = 100000
    plot_count = min(int(episode_count / 100), 1000)
    avgs = []

    sv = SmartVac()

    episode_rewards = np.zeros(episode_count)

    for i_episode in range(episode_count):
        done = False
        totalReward = 0

        if i_episode >= plot_count and (i_episode % plot_count == 0):
            avg = np.average(episode_rewards[i_episode - plot_count:i_episode])
            avgs.append(avg)

            print('.', end='', flush=True)
            if len(avgs) % 100 == 0:
                print(i_episode)

        obs = sv.reset()

        while not done:
            obs, reward, done = sv.step(random.choice(list(range(4))))
            totalReward += reward

        episode_rewards[i_episode] = totalReward

    plt.figure(1)
    plt.plot(avgs)
    plt.title(f'Average Reward in {episode_count} episodes')
    plt.xlabel(f'index')
    plt.ylabel(f'Average Reward per {plot_count} episodes')
    plt.show()

    print('')
    print('Average:', np.mean(avgs))
    print(f'Best {plot_count} Average:', np.max(avgs))
