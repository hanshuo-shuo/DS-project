import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import gym_myhunting
import matplotlib.pyplot as plt
import copy
import random
# import wandb
#
# PROJECT_NAME = 'DQN-Hunting'
# run = wandb.init(project=PROJECT_NAME, resume=False)
# hyper-parameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100
""" Epsilon """
epsilon_begin = 1.0
epsilon_end = 0.1
epsilon_decay = 10000

env = gym.make("hunting-v0")
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

device = torch.device('cpu')

filename_model_p_eval = "c1ae3/p_eval.pth"
filename_model_e_eval = "c1ae3/e_eval.pth"
filename_model_p_target = "c1ae3/p_target.pth"
filename_model_e_target = "c1ae3/e_target.pth"

class Net(nn.Module):
    """docstring for Net"""

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, NUM_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob


class DQN():
    """docstring for DQN"""

    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    # def choose_action(self, state):
    #     state = torch.unsqueeze(torch.FloatTensor(state), 0)  # get a 1D array
    #     if np.random.randn() <= EPISILO:  # greedy policy
    #         action_value = self.eval_net.forward(state)
    #         action = torch.max(action_value, 1)[1].data.numpy()
    #         action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
    #     else:  # random policy
    #         action = np.random.randint(0, NUM_ACTIONS)
    #         action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
    #     return action

    def choose_action(self, obs, epsilon):
        if random.random() < epsilon:
            action = random.randrange(NUM_ACTIONS)
        else:
            with torch.no_grad():
                action = torch.argmax(self.eval_net.forward(obs.unsqueeze(0))).item()
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):

        # update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        # q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    dqn_p = DQN()
    dqn_e = DQN()
    print("Collecting Experience....")
    step = 0
    reward_list_p = []
    reward_list_e = []
    for i in range(100000):
        step_episode = 0
        obs_p, obs_e = env.my_reset()

        obs_p, obs_e = torch.Tensor(obs_p), torch.Tensor(obs_e)
        done = False
        total_reward_p = 0
        total_reward_e = 0
        epsilon_func = lambda step: max(epsilon_end,
                                        epsilon_begin - (epsilon_begin - epsilon_end) * (step / epsilon_decay))
        while not done:
            action_p = dqn_p.choose_action(obs_p.float().to(device), epsilon_func(step))
            action_e = dqn_e.choose_action(obs_e.float().to(device), epsilon_func(step))
            next_obs_p, next_obs_e, reward_p, reward_e, done = env.my_step(action_p, action_e, step_episode)
            next_obs_p, next_obs_e = torch.Tensor(next_obs_p), torch.Tensor(next_obs_e)

            total_reward_p += reward_p
            total_reward_e = reward_e

            dqn_p.store_transition(obs_p.squeeze(0), action_p, reward_p, next_obs_p.squeeze(0))
            dqn_e.store_transition(obs_e.squeeze(0), action_e, reward_e, next_obs_e.squeeze(0))

            if dqn_p.memory_counter >= MEMORY_CAPACITY:
                dqn_p.learn()
            if dqn_e.memory_counter >= MEMORY_CAPACITY:
                dqn_e.learn()
            obs_p = next_obs_p
            obs_e = next_obs_e
            step_episode += 1
            step += 1
        reward_list_p.append(total_reward_p)
        reward_list_e.append(total_reward_e)

        ## save model
        if (i + 1) % 10 == 0:
            torch.save(dqn_p.eval_net.state_dict(), filename_model_p_eval)
            torch.save(dqn_p.target_net.state_dict(), filename_model_p_target)
            torch.save(dqn_e.eval_net.state_dict(), filename_model_e_eval)
            torch.save(dqn_e.target_net.state_dict(), filename_model_e_target)

        ## visulalization
        if (i + 1) % 50 == 0:
            wandb.log({'Episode_step': step_episode + 1, "Reward_p": total_reward_p, "Reward_e": total_reward_e})


if __name__ == '__main__':
    main()