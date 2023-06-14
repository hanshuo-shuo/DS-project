import torch
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn as nn
import gym
import gym_myhunting
import matplotlib.pyplot as plt
from DQN import DQN
device = torch.device('cpu')

## envs
env = gym.make("hunting-v0")
""" Epsilon """
epsilon_begin = 1.0
epsilon_end = 0.1
epsilon_decay = 10000

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

##reload the network
dqn_p = DQN()
dqn_e = DQN()
dqn_p.eval_net.load_state_dict(torch.load("c1ae3/p_eval.pth"))
dqn_e.eval_net.load_state_dict(torch.load("c1ae3/e_eval.pth"))
dqn_p.target_net.load_state_dict(torch.load("c1ae3/p_target.pth"))
dqn_e.target_net.load_state_dict(torch.load("c1ae3/e_target.pth"))
epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

# run the env
step = 10000
step_episode = 0
obs_p, obs_e = env.my_reset()
obs_p, obs_e = torch.Tensor(obs_p), torch.Tensor(obs_e)
done = False
predator_position_x = []
predator_position_y = []
prey_position_x = []
prey_position_y = []
while not done:
    action_p = dqn_p.choose_action(obs_p.float().to(device), epsilon_func(step))
    action_e = dqn_e.choose_action(obs_e.float().to(device), epsilon_func(step))

    next_obs_p, next_obs_e, reward_p, reward_e, done = env.my_step(action_p, action_e, step_episode)
    next_obs_p, next_obs_e = torch.Tensor(next_obs_p), torch.Tensor(next_obs_e)

    predator_position_x.append(obs_p[0][0])
    predator_position_y.append(obs_p[0][1])
    prey_position_x.append(obs_p[0][4])
    prey_position_y.append(obs_p[0][5])

    obs_p = next_obs_p
    obs_e = next_obs_e

    step_episode += 1

print(step_episode)




if len(predator_position_x) > 2:
    for k in range(len(predator_position_x)):
        fig, ax = plt.subplots()
        ax.axis([-1.5, 1.5, -1.5, 1.5])
        ## block on the top
        ax.plot([-0.5, -0.1], [0.5, 0.5], color='red', linewidth=1)
        ax.plot([-0.5, -0.1], [0.75, 0.75], color='red', linewidth=1)
        ax.plot([-0.5, -0.5], [0.5, 0.75], color='red', linewidth=1)
        ax.plot([-0.1, -0.1], [0.5, 0.75], color='red', linewidth=1)
        ax.plot([-0.5, -0.1], [0.75, 0.5], color='red', linewidth=2)
        ax.plot([-0.1, -0.5], [0.75, 0.5], color='red', linewidth=2)

        ## block bottom
        ax.plot([-0.5, -0.1], [-0.5, -0.5], color='red', linewidth=1)
        ax.plot([-0.5, -0.1], [-0.75, -0.75], color='red', linewidth=1)
        ax.plot([-0.5, -0.5], [-0.5, -0.75], color='red', linewidth=1)
        ax.plot([-0.1, -0.1], [-0.5, -0.75], color='red', linewidth=1)
        ax.plot([-0.5, -0.1], [-0.75, -0.5], color='red', linewidth=2)
        ax.plot([-0.1, -0.5], [-0.75, -0.5], color='red', linewidth=2)

        ## block on the right
        ax.plot([0.3, 0.3], [0.3, -0.3], color='red', linewidth=1)
        ax.plot([0.5, 0.5], [0.3, -0.3], color='red', linewidth=1)
        ax.plot([0.3, 0.5], [0.3, 0.3], color='red', linewidth=1)
        ax.plot([0.3, 0.5], [-0.3, -0.3], color='red', linewidth=1)
        ax.plot([0.3, 0.5], [0.3, -0.3], color='red', linewidth=2)
        ax.plot([0.3, 0.5], [-0.3, 0.3], color='red', linewidth=2)

        ## destination for prey
        ax.plot([0.75, 0.75], [0.25, -0.25], color='green', linewidth=2)
        ax.plot([0.75, 1], [0.25, 0.25], color='green', linewidth=2)
        ax.plot([0.75, 1], [-0.25, -0.25], color='green', linewidth=2)

        ax.scatter(predator_position_x[k], predator_position_y[k], color="red", s=50, marker="^")
        ax.scatter(prey_position_x[k], prey_position_y[k], color="black", s=50, marker="s")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Remove the tick marks
        ax.tick_params(axis='both', which='both', length=0)
        # Set the background color to white
        ax.set_facecolor('white')
        plt.savefig(f"pic/frame_{k}.png")
        plt.close()

### first transfor to frames then video
frames = []

for frame in range(len(predator_position_x)):
    frames.append(Image.open(f"pic/frame_{frame}.png"))
frames[0].save("gif/testOMGH1.gif", format='GIF', append_images=frames[1:], save_all=True)