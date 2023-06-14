import sys, os
import gym
from gym import Env
from gym.spaces import Box, Discrete
sys.path.append(os.pardir)
import numpy as np
from common.util import *
import random


class mychase(Env):
    def __init__(self, speed_pursuer = 3.0, speed_evader = 3.0, mass_pursuer=1, mass_evader=1, damping=0.25, dt=0.1, max_step=300):
        self.speed_p = speed_pursuer
        self.speed_e = speed_evader
        self.mass_p = mass_pursuer
        self.mass_e = mass_evader
        self.damping = damping
        self.dt = dt
        self.max_step = max_step
        self.action_space = Discrete(13)
        self.observation_space = Box(low=np.array([-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]), high=np.array([1,1,1,1,1,1,1,1,1,1]))
        # self.action_space = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
        # self.observation_space = self.observation_space()


    def my_reset(self):
        self.pos_p = np.random.uniform(-0.25, 0.25, 2)
        self.vel_p = np.zeros(2)
        self.pos_e = np.array([-0.9, random.uniform(-0.5, 0.5)])
        self.vel_e = np.zeros(2)

        obs_p = get_obs_p(self.pos_p, self.vel_p, self.pos_e, self.vel_e)
        obs_e = get_obs_e(self.pos_e, self.vel_e, self.pos_p, self.vel_p)

        return obs_p, obs_e

    def my_step(self, action_p, action_e, num_step):
        abs_u_p = get_abs_u(action_p, self.pos_p, self.pos_e)
        next_pos_p, next_vel_p = get_next_own_state(self.pos_p, self.vel_p, abs_u_p, \
                                                    self.mass_p, self.speed_p, self.damping, self.dt)

        abs_u_e = get_abs_u(action_e, self.pos_e, self.pos_p)
        next_pos_e, next_vel_e = get_next_own_state(self.pos_e, self.vel_e, abs_u_e, \
                                                    self.mass_e, self.speed_e, self.damping, self.dt)

        next_obs_p = get_obs_p(next_pos_p, next_vel_p, next_pos_e, next_vel_e)
        next_obs_e = get_obs_e(next_pos_e, next_vel_e, next_pos_p, next_vel_p)

        ## if p hits the wall
        if next_pos_p[0] > 1:
            next_pos_p[0] = 0.9
        elif next_pos_p[1] > 1:
            next_pos_p[1] = 0.9
        elif next_pos_p[0] < -1:
            next_pos_p[0] = -0.9
        elif next_pos_p[1] < -1:
            next_pos_p[1] = -0.9

        ## if p hits the block above
        if next_pos_p[0] < -0.1 and next_pos_p[0] > -0.5 and next_pos_p[1] < 0.75 and next_pos_p[1] > 0.5:
            dleft = next_pos_p[0] + 0.5
            dright = -0.1 - next_pos_p[0]
            dup = 0.75 - next_pos_p[1]
            ddown = next_pos_p[1] - 0.25
            min_dist = min(dleft, dright, dup, ddown)
            if min_dist == dleft:
                next_pos_p[0] = - 0.55
            elif min_dist == dright:
                next_pos_p[0] = -0.05
            elif min_dist == dup:
                next_pos_p[1] = 0.8
            elif min_dist == ddown:
                next_pos_p[1] = 0.45

        ## if p hits the block below
        if next_pos_p[0] < -0.1 and next_pos_p[0] > -0.5 and next_pos_p[1] > -0.75 and next_pos_p[1] < -0.5:
            dleft = next_pos_p[0] + 0.5
            dright = -0.1 - next_pos_p[0]
            dup = -0.5 - next_pos_p[1]
            ddown = next_pos_p[1] + 0.75
            min_dist = min(dleft, dright, dup, ddown)
            if min_dist == dleft:
                next_pos_p[0] = - 0.55
            elif min_dist == dright:
                next_pos_p[0] = -0.05
            elif min_dist == dup:
                next_pos_p[1] = -0.45
            elif min_dist == ddown:
                next_pos_p[1] = -0.8

        ## if p hits the block on the right
        if next_pos_p[0] > 0.3 and next_pos_p[0] < 0.5 and abs(next_pos_p[1] < 0.3):
            dleft = next_pos_p[0] - 0.3
            dright = 0.5 - next_pos_p[0]
            dup = 0.3 - next_pos_p[1]
            ddown = next_pos_p[1] + 0.3
            min_dist = min(dleft, dright, dup, ddown)
            if min_dist == dleft:
                next_pos_p[0] = 0.25
            elif min_dist == dright:
                next_pos_p[0] = 0.55
            elif min_dist == dup:
                next_pos_p[1] = 0.35
            elif min_dist == ddown:
                next_pos_p[1] = -0.35


        ## if e hits the wall
        if next_pos_e[0] > 1:
            next_pos_e[0] = 0.9
        elif next_pos_e[1] > 1:
            next_pos_e[1] = 0.9
        elif next_pos_e[0] < -1:
            next_pos_e[0] = -0.9
        elif next_pos_e[1] < -1:
            next_pos_e[1] = -0.9

        ## if e hits the block below
        if next_pos_e[0] < -0.1 and next_pos_e[0] > -0.5 and next_pos_e[1] > -0.75 and next_pos_e[1] < -0.5:
            dleft = next_pos_e[0] + 0.5
            dright = -0.1 - next_pos_e[0]
            dup = -0.5 - next_pos_e[1]
            ddown = next_pos_e[1] + 0.75
            min_dist = min(dleft, dright, dup, ddown)
            if min_dist == dleft:
                next_pos_e[0] = - 0.55
            elif min_dist == dright:
                next_pos_e[0] = -0.05
            elif min_dist == dup:
                next_pos_e[1] = -0.45
            elif min_dist == ddown:
                next_pos_e[1] = -0.8

        ## if e hits the block above
        if next_pos_e[0] < -0.1 and next_pos_e[0] > -0.5 and next_pos_e[1] < 0.75 and next_pos_e[1] > 0.5:
            dleft = next_pos_e[0] + 0.5
            dright = -0.1 - next_pos_e[0]
            dup = 0.75 - next_pos_e[1]
            ddown = next_pos_e[1] - 0.25
            min_dist = min(dleft, dright, dup, ddown)
            if min_dist == dleft:
                next_pos_e[0] = - 0.55
            elif min_dist == dright:
                next_pos_e[0] = -0.05
            elif min_dist == dup:
                next_pos_e[1] = 0.8
            elif min_dist == ddown:
                next_pos_e[1] = 0.45


        ## if e hits the block on the right
        if next_pos_e[0] > 0.3 and next_pos_e[0] < 0.5 and abs(next_pos_e[1] < 0.3):
            dleft = next_pos_e[0] - 0.3
            dright = 0.5 - next_pos_e[0]
            dup = 0.3 - next_pos_e[1]
            ddown = next_pos_e[1] + 0.3
            min_dist = min(dleft, dright, dup, ddown)
            if min_dist == dleft:
                next_pos_e[0] = 0.25
            elif min_dist == dright:
                next_pos_e[0] = 0.55
            elif min_dist == dup:
                next_pos_e[1] = 0.35
            elif min_dist == ddown:
                next_pos_e[1] = -0.35


        reward_p = get_reward_pursuer(next_pos_p, next_pos_e)
        reward_e = get_reward_evader(next_pos_e, next_pos_p)

        done = get_done(next_pos_e, next_pos_p, num_step, self.max_step)

        self.pos_p = next_pos_p
        self.vel_p = next_vel_p
        self.pos_e = next_pos_e
        self.vel_e = next_vel_e


        return next_obs_p, next_obs_e, reward_p, reward_e, done

    def close(self):
        pass

    def seed(self, seed=None):
        pass

    def render(self):
        pass





def get_obs_p(pos_p, vel_p, pos_e, vel_e):
    sub_pos_adv = get_sub_pos(pos_p, pos_e)
    sub_vel_own = get_sub_vel(pos_p, pos_e, vel_p)
    sub_vel_adv = get_sub_vel(pos_p, pos_e, vel_e)
    obs_p = np.concatenate([pos_p] + [sub_vel_own] + \
                           [pos_e] + [sub_pos_adv] + [sub_vel_adv]).reshape(1, 10)

    return obs_p


def get_obs_e(pos_e, vel_e, pos_p, vel_p):
    sub_pos_adv = get_sub_pos(pos_e, pos_p)
    sub_vel_own = get_sub_vel(pos_e, pos_p, vel_e)
    sub_vel_adv = get_sub_vel(pos_e, pos_p, vel_p)
    obs_e = np.concatenate([pos_e] + [sub_vel_own] + \
                           [pos_p] + [sub_pos_adv] + [sub_vel_adv]).reshape(1, 10)

    return obs_e


def get_reward_pursuer(abs_pos_own, abs_pos_adv):
    dist = get_dist(abs_pos_own, abs_pos_adv)
    reward = 0
    if dist < 0.1:
        reward = 10
    return reward


def get_reward_evader(abs_pos_own, abs_pos_adv):
    dist = get_dist(abs_pos_own, abs_pos_adv)
    reward = 0
    des = np.array([1, 0])
    dist_to_des = get_dist(abs_pos_own, des)
    reward = 2 - dist_to_des
    if dist < 0.1:
        reward -= 10
    elif abs_pos_own[0] > 0.75 and abs_pos_own[0] < 1 and abs(abs_pos_own[1]) < 0.25:
        reward += 15
    return reward


def get_done(abs_pos_own, abs_pos_adv, num_step, max_step):
    ## here abs_pos_own needs to be the escaper
    dist = get_dist(abs_pos_own, abs_pos_adv)
    if dist < 0.1 or num_step > max_step:
        done = True
    elif abs_pos_own[0] > 0.75 and abs_pos_own[0] < 1 and abs(abs_pos_own[1]) < 0.25:
        done = True
    else:
        done = False
    return done