# myhunting-v0

Go to gym_myhunting file:
```commandline
pip install -e .
```
The -e means that the package is installed in "editable" mode. This means that the package is installed locally, and any changes to the original package will be reflected in the environment.
- Random goal generate
- A point trying to reach a target goal in each round

```python
import gym
import gym_myhunting
env = gym.make('hunting-v0')
```

```commandline
python sac_continuous_action.py --env-id mymaze-v0 --track --capture-video
```

# Building a Reinforcement Learning Environment using OpenAI Gym

```python
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
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
        
    def step(self, action):
        self.state += action -1 
        self.shower_length -= 1 
        # Calculating the reward
        if self.state >=37 and self.state <=39: 
            reward =1 
        else: 
            reward = -1 
        # Checking if shower is done
        if self.shower_length <= 0: 
            done = True
        else:
            done = False
        # Setting the placeholder for info
        info = {}
        # Returning the step information
        return self.state, reward, done, info
    
    def render(self):
        pass
        #visulize
    
    def reset(self):
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60 
        return self.state
    
    def close(self):
        pass
    
    def seed(self, seed=None):
        pass

```