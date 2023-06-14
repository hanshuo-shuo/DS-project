import gym
import d4rl


def goal_generate(n=10000):
    env = gym.make('maze2d-umaze-v1')
    env.reset()
    env.step(env.action_space.sample())
    s_buffer = []
    for i in range(n):
        action = env.action_space.sample()
        next_obs, reward, terminated, info = env.step(action)
        s_buffer.append(next_obs[0:2])
    return s_buffer



def main():
    data = goal_generate()
    for goal in data:
        print(goal)

if __name__ == "__main__":
    main()
