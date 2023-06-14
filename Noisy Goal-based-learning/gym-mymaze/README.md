# mymaze-v0

Go to gym-mymaze file:
```commandline
pip install -e .
```
The -e means that the package is installed in "editable" mode. This means that the package is installed locally, and any changes to the original package will be reflected in the environment.
- Random goal generate
- A point trying to reach a target goal in each round

```python
import gym
import gym_mymaze
env = gym.make('mymaze-v0')
```

```commandline
python sac_continuous_action.py --env-id mymaze-v0 --track --capture-video
```