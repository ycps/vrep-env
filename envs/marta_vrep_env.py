
import vrep_env
import vrep # vrep.sim_handle_parent

import gym
from gym import spaces

import numpy as np

class MartaVrepEnv(vrep_env.VrepEnv):
	metadata = {
		'render.modes': [],
	}
	def __init__(
		self,
		server_addr='127.0.0.1',
		server_port=19997,
		#scene_path='/home/VREP/scenes/Martabot_v1.2.1.ttt',
		#scene_path='/home/VREP/scenes/Martabot_v1.2.1_lane.ttt',
		scene_path='/home/VREP/scenes/Martabot_v1.2.1_lane_cam.ttt',
	):
		vrep_env.VrepEnv.__init__(
			self,
			server_addr,
			server_port,
			scene_path,
		)
		
		# All Joints
		joint_names = [
			#               'HeadPitch',
			#               'NeckYaw',
			'LeftShoulderPitch', 'RightShoulderPitch',
			'LeftShoulderRoll',  'RightShoulderRoll',
			#'LeftElbowPitch',    'RightElbowPitch',
			               'HipRoll',
			               'HipPitch',
			               'HipYaw',
			'LeftLegYaw',        'RightLegYaw',
			'LeftPelvisPitch',   'RightPelvisPitch',
			'LeftPelvisRoll',    'RightPelvisRoll',
			'LeftKneePitch',     'RightKneePitch',
			'LeftAnkleRoll',     'RightAnkleRoll',
			'LeftAnklePitch',    'RightAnklePitch',
			'LeftFingersPitch',  'RightFingersPitch',
		]
		# Some shapes
		shape_names = [
			#               'Head',
			#               'Neck',
			               'Chest',
			'LeftShoulderLink', 'RightShoulderLink',
			#'LeftForearm',      'RightForearm',
			#'LeftArm',          'RightArm',
			#'LeftHand',         'RightHand',
			               'PelvisMiddle',
			               'PelvisLink',
			               'PelvisLower',
			'LeftPelvisLink',   'RightPelvisLink',
			'LeftThighMotors',  'RightThighMotors',
			#'LeftThigh',        'RightThigh',
			#'LeftCalf',         'RightCalf',
			'LeftCalfMotors',   'RightCalfMotors',
			#'LeftAnkle',        'RightAnkle',
			'LeftFingers',      'RightFingers'
		]
		
		# Getting object handles
		
		# Meta
		#self.webcam = self.get_object_handle('webcam')
		
		# Actuators
		self.oh_joint = list(map(self.get_object_handle, joint_names))
		# Shapes
		self.oh_shape = list(map(self.get_object_handle, shape_names))
		
		# One action per joint
		num_act = len(self.oh_joint) ;
		# Multiple dimensions per shape
		num_obs = ((len(self.oh_shape)+(1) )*3*(1) )+0 ;
		
		self.joints_max_velocity = 3.0
		act = np.array( [self.joints_max_velocity] * num_act )
		obs = np.array(          [np.inf]          * num_obs )
		
		self.action_space      = spaces.Box(-act,act)
		self.observation_space = spaces.Box(-obs,obs)
		
		print('MartaVrepEnv: initialized')
	
	def _make_observation(self):
			
		lst_o = [];
		
		# Include chest absolute position in observation
		pos                = self.obj_get_position(self.oh_shape[0])
		#lin_vel , ang_vel = self.obj_get_velocity(self.oh_shape[0])
		lst_o += pos ;
		#lst_o += lin_vel ;
		
		# Include shapes relative position in observation
		for i_oh in self.oh_shape:
			rel_pos = self.obj_get_position(i_oh,relative_to=vrep.sim_handle_parent)
			lst_o += rel_pos ;
			#lin_vel , ang_vel = self.obj_get_velocity(oh_i)
			#lst_o += ang_vel ;
			#lst_o += lin_vel ;
		
		self.observation = np.array(lst_o).astype('float32');
	
	def _make_action(self, a):
		for i_oh, i_a in zip(self.oh_joint, a):
			self.obj_set_velocity(i_oh, i_a)
	
	def _step(self, action):
		# actions = np.clip(actions,-self.joints_max_velocity, self.joints_max_velocity)
		assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
		
		# Actuate
		self._make_action(action)
		# Step
		self.step_simulation()
		# Observe
		self._make_observation()
		
		# Reward
		chest_pos_x  = self.observation[0] # front/back
		#chest_pos_y = self.observation[1] # left/right
		chest_pos_z  = self.observation[2] # up/down
		#nrm_action  = np.linalg.norm(actions)
		#nrm_obsrvt  = np.linalg.norm(self.observation[3:])
		#r_regul     = -(nrm_action**2)
		r_alive = 3.0
		
		reward = (1.0)*(r_alive) +(4.0)*(chest_pos_x) +(1.0)*(chest_pos_z)
		
		# Early stop
		#done = False
		stand_threshold = 0.20
		done = (chest_pos_z < stand_threshold)
		
		return self.observation, reward, done, {}
	
	def _reset(self):
		if self.sim_running:
			self.stop_simulation()
		
		self.start_simulation()
		self._make_observation()
		
		return self.observation
	
	def _render(self, mode='human', close=False):
		pass
	
	def _seed(self, seed=None):
		return []
	
def main(args):
	env = MartaVrepEnv()
	for i_episode in range(8):
		observation = env.reset()
		total_reward = 0
		for t in range(256):
			# (do not use this on a real robot)
			action = env.action_space.sample()
			observation, reward, done, _ = env.step(action)
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
	
