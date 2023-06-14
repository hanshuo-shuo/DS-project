import random
import gym
import d4rl
import numpy as np

class mymaze(gym.Env):
    def __init__(self, env_id = "maze2d-umaze-v1", corruption = True):
        env = gym.make(env_id)
        env.reset()
        self.maze_base = env
        self.past_visited_states = []
        # initialization
        obs, reward, terminated, info = self.maze_base.step(self.maze_base.action_space.sample())
        self.current_goal = obs
        self.corruption = corruption
        self.action_space = env.action_space
        self.observation_space = env.observation_space


    def get_reward(self, obs):
        # Two kinds of distance
        #d = np.linalg.norm(obs - self.current_goal)
        d = np.linalg.norm(obs[0:2] - self.current_goal[0:2])
        if d < 0.1:
            return 1
        return 0

    def step(self, action):
        obs, reward, terminated, info = self.maze_base.step(action)
        self.past_visited_states.append(obs)
        reward = self.get_reward(obs)
        if self.corruption is True:
            if (abs(obs[0])<1 and abs(obs[1])<1):
                obs = np.array([random.uniform(0, 1), random.uniform(0, 1),
                       random.uniform(-1, 1), random.uniform(-1, 1)])
                #obs = [obs[0]+random.uniform(-0.2,0.2),obs[1]+random.uniform(-0.2,0.2),obs[2]+random.uniform(-0.2,0.2),obs[3]+random.uniform(-0.2,0.2)]

        return obs, reward, terminated, info

    def setgoal_eve(self):
        obs = self.maze_base.reset()
        self.current_goal = obs
        # else:
            # self.current_goal = random.choice(self.past_visited_states)
        return self.current_goal

    def setgoal_cor(self):
        self.current_goal = np.array([random.uniform(0, 1), random.uniform(0, 1),
                       random.uniform(-1, 1), random.uniform(-1, 1)])
        return self.current_goal


    def reset(self):
        # set goal everywhere
        # self.current_goal = self.setgoal_eve()
        # set goal only from corruption zone
        self.current_goal = self.setgoal_cor()
        print(self.current_goal)
        return self.maze_base.reset()


    def render(self):
        return self.maze_base.render()


    def close(self):
        return self.maze_base.close()


def main():
    goal = mymaze()
    obs = goal.reset()
    for i in range(10000):
        action = obs[2:4]
        obs, reward, terminated, info = goal.step(action)
        goal.render()
    goal.close()


if __name__ == "__main__":
    main()