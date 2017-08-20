
import vrep_env

import gym
from gym import spaces

import numpy as np

# Used for visualization only
#import cv2
#import scipy.misc

class CartPoleVrepEnv(vrep_env.VrepEnv):
	metadata = {
		'render.modes': ['human', 'rgb_array'],
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		scene_path='/home/VREP/scenes/cart_pole_up.ttt',
	):
		vrep_env.VrepEnv.__init__(
			self,
			server_addr,
			server_port,
			scene_path,
		)
		
		# Actuators
		self.slider = self.get_object_handle('slider')
		
		# Shapes
		self.cart = self.get_object_handle('cart')
		self.mass = self.get_object_handle('mass')
		
		# Meta
		self.webcam = self.get_object_handle('webcam')
		
		# cart+mass, (pos,vel), 2 dimensions each
		obs = np.array([np.inf]*(2*2+2*2))
		
		# Pick continuous/discrete action space
		self.meta_continous = False
		
		if self.meta_continous:
			act = np.array([2.]*(2))
			self.action_space     = gym.spaces.Box(-act,act)
		else:
			self.action_space      = spaces.Discrete(2)
		self.observation_space = gym.spaces.Box(-obs,obs)
		
		print('CartPoleVrepEnv: initialized')
	
	def _make_observation(self):
		cart_pos = self.obj_get_position(self.cart)
		mass_pos = self.obj_get_position(self.mass)
		
		cart_vel,cart_angvel = self.obj_get_velocity(self.cart)
		mass_vel,mass_angvel = self.obj_get_velocity(self.mass)
		
		# Only use x and z, ignoring y
		self.observation = np.array([
			cart_pos[0],cart_vel[0], mass_pos[0],mass_vel[0],
			cart_pos[2],cart_vel[2], mass_pos[2],mass_vel[2]
		]).astype('float32')
	
	def _step(self, action):
		# actions = np.clip(actions, -2,2)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		
		if self.meta_continous:
			v = action[0]
		else:
			v = (+1.2) if action==1 else (-1.2)
		
		# Actuate
		self.obj_set_velocity(self.slider, v)
		
		# Step
		self.step_simulation()
		
		# Observe
		self._make_observation()
		
		# Reward
		r_alive = 1.0
		r_regul = -(v**2)
		mass_height = self.observation[6]
		r_objt = (mass_height)
		
		reward = (64)*r_objt + (0.05)*r_regul + (0.05)*r_alive
		
		# Early stop
		#done = False
		tolerable_height = +4.5000e-01
		done = (mass_height < tolerable_height)
		
		return self.observation, reward, done, {}
		
	def _reset(self):
		if self.sim_running:
			self.stop_simulation()
		
		self.start_simulation()
		self._make_observation()
		
		return self.observation
	
	def _render(self, mode='human', close=False):
		if close:
			#cv2.destroyAllWindows()
			return
		img = self.obj_get_vision_image(self.webcam)
		if mode == 'human':
			#cv2.imshow('render',img)
			#cv2.waitKey(1)
			pass
		elif mode == 'rgb_array':
			return img
		
	def _seed(self, seed=None):
		return []
	

def main(args):
	env = CartPoleVrepEnv()
	for i_episode in range(8):
		observation = env.reset()
		total_reward = 0
		for t in range(256):
			# feedback
			#frame = env.render('rgb_array')
			#scipy.misc.imsave('/tmp/clip/f-{ie}-{it}.png'.format(ie=i_episode,it=t), frame)
			#print(observation)
			
			# pick a random action
			action = env.action_space.sample()
			observation, reward, done, info = env.step(action)
			total_reward += reward
			if done:
				print("Episode finished after {} timesteps".format(t+1))
				print("Total reward: {}".format(total_reward))
				break
	env.close()
	return 0

if __name__ == '__main__':
	import sys
	sys.exit(main(sys.argv))
