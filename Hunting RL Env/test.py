import torch
from torch import nn, optim
from common.network import DuelingNetwork
from common.replay import PrioritizedReplayBuffer
from common.trainer import Trainer
from common.hparameter import *
from c1ae.modified_env import My_chase_1
from c1ae.chase1_and_escape import Chase1AndEscape
import matplotlib.pyplot as plt
from PIL import Image
from gif_to_array import readGif

device = torch.device('cpu')

net_p = DuelingNetwork(10, 13).to(device)
target_net_p = DuelingNetwork(10, 13).to(device)
optimizer_p = optim.Adam(net_p.parameters(), lr=learning_rate)

net_e = DuelingNetwork(10, 13).to(device)
target_net_e = DuelingNetwork(10, 13).to(device)
optimizer_e = optim.Adam(net_e.parameters(), lr=learning_rate)

loss_func = nn.SmoothL1Loss(reduction='none')

net_p.load_state_dict(torch.load("model/c1ae3/p_3.0.pth"))
#net_p.eval()
net_e.load_state_dict(torch.load("model/c1ae3/e_3.0.pth"))
#net_e.eval()


env = My_chase_1(speed_pursuer=3.0, speed_evader=3, max_step=3000)
epsilon_func = lambda step: max(epsilon_end, epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))
step = 0   

replay_buffer_p = PrioritizedReplayBuffer(buffer_size)
replay_buffer_e = PrioritizedReplayBuffer(buffer_size)
beta_func = lambda step: min(beta_end, beta_begin + (beta_end - beta_begin) * (step / beta_decay))

trainer_p = Trainer(net_p, target_net_p, optimizer_p, loss_func, replay_buffer_p, gamma, device)
trainer_e = Trainer(net_e, target_net_e, optimizer_e, loss_func, replay_buffer_e, gamma, device)



for episode in range(1):
    step_episode = 0
    step = 0
    obs_p, obs_e = env.reset()
    obs_p, obs_e = torch.Tensor(obs_p), torch.Tensor(obs_e)
    done = False
    total_reward_p = 0
    total_reward_e = 0
    predator_position_x = []
    predator_position_y = []
    prey_position_x = []
    prey_position_y = []
    while not done:
        action_p = net_p.act(obs_p.float().to(device), epsilon_func(step))
        action_e = net_e.act(obs_e.float().to(device), epsilon_func(step))        
        
        next_obs_p, next_obs_e, reward_p, reward_e, done = env.step(action_p, action_e, step_episode)        
        next_obs_p, next_obs_e = torch.Tensor(next_obs_p), torch.Tensor(next_obs_e)

        total_reward_p += reward_p
        total_reward_e = reward_e

        replay_buffer_p.push([obs_p, action_p, reward_p, next_obs_p, done])
        replay_buffer_e.push([obs_e, action_e, reward_e, next_obs_e, done])

        obs_p = next_obs_p
        obs_e = next_obs_e

        predator_position_x.append(obs_p[0][0])
        predator_position_y.append(obs_p[0][1])
        prey_position_x.append(obs_p[0][4])
        prey_position_y.append(obs_p[0][5])

        if len(replay_buffer_p) >= initial_buffer_size:
            trainer_p.update(batch_size, beta_func(step))
            trainer_e.update(batch_size, beta_func(step))

        if (step + 1) % target_update_interval == 0:
            target_net_p.load_state_dict(net_p.state_dict())
            target_net_e.load_state_dict(net_e.state_dict())

        step_episode += 1
        step += 1
    #plt.plot(predator_position_x,predator_position_y)
    #plt.show()
    #print(len(predator_position_x))

print(step_episode)

if len(predator_position_x) > 2:
    for k in range(len(predator_position_x)):
        fig, ax = plt.subplots()
        ax.axis([-1, 1, -1, 1])
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


        ax.scatter(predator_position_x[k], predator_position_y[k], color="red", s = 50, marker= "^")
        ax.scatter(prey_position_x[k], prey_position_y[k], color="black", s = 50, marker= "s")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # Remove the tick marks
        ax.tick_params(axis='both', which='both', length=0)
        # Set the background color to white
        ax.set_facecolor('white')
        plt.savefig(f"./pic/frame_{k}.png")
        plt.close()

### first transfor to frames then video
frames = []

for frame in range(len(predator_position_x)):
    frames.append(Image.open(f"pic/frame_{frame}.png"))
frames[0].save("gif/testOMGH1.gif", format='GIF', append_images=frames[1:], save_all=True)
