import torch
from common.network import DuelingNetwork
from common.hparameter import *
from c1ae.chase1_and_escape import Chase1AndEscape
import matplotlib.pyplot as plt
from PIL import Image

device = torch.device('cpu')

## load the trained model
net_p = DuelingNetwork(10, 13).to(device)
target_net_p = DuelingNetwork(10, 13).to(device)
net_p.load_state_dict(torch.load("model/c1ae/p_2.4.pth"))
net_p.eval()

net_e = DuelingNetwork(10, 13).to(device)
target_net_e = DuelingNetwork(10, 13).to(device)
net_e.load_state_dict(torch.load("model/c1ae/e_2.4.pth"))
net_e.eval()

env = Chase1AndEscape(speed_pursuer=2.4, speed_evader=3, max_step=300)
epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))

step = 0

def gif_generate(n = 10, resize = True):
    max_step = 100
    real_label = 2885
    for i in range(n):
        eps = 0
        step_episode = 0  # can be any value
        obs_p, obs_e = env.reset()
        obs_p, obs_e = torch.Tensor(obs_p), torch.Tensor(obs_e)
        done = False
        total_reward_p = 0
        total_reward_e = 0
        predator_position_x = []
        predator_position_y = []
        prey_position_x = []
        prey_position_y = []
        while not done and (eps < max_step):
            action_p = net_p.act(obs_p.float().to(device), epsilon_func(step))
            action_e = net_e.act(obs_e.float().to(device), epsilon_func(step))

            next_obs_p, next_obs_e, reward_p, reward_e, done = env.step(action_p, action_e, step_episode)
            next_obs_p, next_obs_e = torch.Tensor(next_obs_p), torch.Tensor(next_obs_e)
            total_reward_p += reward_p
            total_reward_e += reward_e

            predator_position_x.append(next_obs_p[0][0])
            predator_position_y.append(next_obs_p[0][1])
            prey_position_x.append(next_obs_p[0][4])
            prey_position_y.append(next_obs_p[0][5])
            eps += 1

        ## plot
        if len(predator_position_x) > 2:
            for k in range(len(predator_position_x)):
                fig, ax = plt.subplots()
                ax.axis([-1, 1, -1, 1])
                ax.scatter(predator_position_x[k], predator_position_y[k], color="black", s=1000, marker='^')
                ax.scatter(prey_position_x[k], prey_position_y[k], color="black", s=1000, marker='s')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # Remove the tick marks
                ax.tick_params(axis='both', which='both', length=0)
                # Set the background color to white
                ax.set_facecolor('white')
                plt.savefig(f"./pic/gif_{real_label}frame_{k}.png")
                plt.close()
                if resize == True:
                    img = Image.open(f"./pic/gif_{real_label}frame_{k}.png")
                    resized_img = img.resize((64, 64))
                    resized_img.save(f"./pic/gif_{real_label}frame_{k}.png")

            frames = []

            for frame in range(len(predator_position_x)):
                frames.append(Image.open(f"pic/gif_{real_label}frame_{frame}.png"))
            frames[0].save(f"gif/test_{real_label}.gif", format='GIF', append_images=frames[1:], save_all=True)
            real_label += 1


if __name__ == "__main__":
    gif_generate(300)

