import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from A2C import ActorNet,CriticNet,GoalMaze
import d4rl
from goal_generate import goal_generate
device = torch.device("cpu")




def main():
    env_id = "maze2d-umaze-v1"
    env = gym.make(env_id)
    state = env.reset()
    next_state, reward, terminated, info = env.step(env.action_space.sample())
    state = env.reset()
    state = torch.FloatTensor(state).to(device)
    agent = GoalMaze(env=env, corruption = False) # Call your agent class.
    actor_losses, critic_losses, episode_rewards_history = [], [], []
    horizon = 100
    current_step = 0
    total_episode_reward = 0

    goal_data = goal_generate(400000)


    for s in goal_data:
        s = np.array(s)
        goal = agent.set_goal(s)
        log_prob, action = agent.select_action(state) # Your agent has a method that returns this.
        next_state, reward, done = agent.step(action) # Your agent also has a method that returns this.
        next_state = torch.FloatTensor(next_state).to(device)
        actor_loss, critic_loss = agent.update_model(state, log_prob, next_state, reward, done)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        state = next_state
        total_episode_reward += reward

        # if episode ends
        if done or (current_step % horizon == 0 and current_step > 1):
            #print("episode terminated")
            state = env.reset()
            state = torch.FloatTensor(state).to(device)
            episode_rewards_history.append(total_episode_reward)
            #plot_fancy(episode_rewards_history)
            print(total_episode_reward)
            total_episode_reward = 0
            current_step += 1
    plt.plot(range(len(episode_rewards_history)), episode_rewards_history, label='Total episode reward')
    plt.legend()
    # plt.plot(range(len(actor_losses)), actor_losses, label='actor_losses')
    # plt.legend()
    # plt.plot(range(len(critic_losses)), critic_losses, label='critic_losses')
    # plt.legend()
    plt.show()
    plt.savefig("A2C_history_no_corruption.png")

if __name__ == "__main__":
    main()