from gym.envs.registration import register

register(
		id='hunting-v0',
		entry_point='gym_myhunting.envs:mychase',
)