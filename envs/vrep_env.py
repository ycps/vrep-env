
import vrep
import gym
import time

import numpy as np

class VrepEnv(gym.Env):
	"""Superclass for V-REP environments.
	"""
	
	def __init__(self,server_addr,server_port,scene_path,):
		# Parameters
		self.server_addr = server_addr
		self.server_port = server_port
		self.scene_path  = scene_path
		
		# Status
		self.cID = -1
		self.connected = False
		self.scene_loaded = (scene_path == None)
		self.sim_running = False
		
		self.connect(server_addr,server_port)
		if not self.scene_loaded:
			self.load_scene(scene_path)
	 
	# internal methods
	
	# Remote API call wrapper
	def RAPI_rc(self, ret_tuple, tolerance=vrep.simx_return_ok):
		istuple = isinstance(ret_tuple, tuple)
		if not istuple:
			ret = ret_tuple
		else:
			ret = ret_tuple[0]
		
		if (ret != vrep.simx_return_ok) and (ret != tolerance):
			raise RuntimeError('Remote API return code: (' + str(ret) + ')')
		
		return ret_tuple[1:] if istuple else None
	
	def connect(self, server_addr, server_port):
		if self.connected:
			raise RuntimeError('Client is already connected.')
		attempts = 0
		max_attempts = 64
		while True:
			self.cID = vrep.simxStart(
				connectionAddress              = server_addr,
				connectionPort                 = server_port,
				waitUntilConnected             = True,
				doNotReconnectOnceDisconnected = True,
				timeOutInMs                    = 1000,
				commThreadCycleInMs            = 0)
			attempts += 1
			if self.cID != -1:
				self.connected = True
				break
			elif attempts > max_attempts:
				raise RuntimeError('Unable to connect to V-REP.')
			else:
				print('Unable to connect to V-REP at ',server_addr,':',server_port,'. Retrying...')
				time.sleep(1);
		
		# Setting up debug signal
		vrep.simxSetIntegerSignal(self.cID,'sig_debug', 1337, vrep.simx_opmode_blocking)
	
	def disconnect(self):
		if not self.connected:
			raise RuntimeError('Client is not even connected.')
		# Clearing debug signal
		vrep.simxClearIntegerSignal(self.cID,'sig_debug', vrep.simx_opmode_blocking)
		vrep.simxFinish(self.cID)
		self.connected = False
	
	def load_scene(self, scene_path):
		if self.scene_loaded:
			raise RuntimeError('Scene is already loaded.')
		self.RAPI_rc(vrep.simxLoadScene(self.cID,scene_path,0, vrep.simx_opmode_blocking))
		self.scene_loaded = True
	
	def close_scene(self):
		if not self.scene_loaded:
			raise RuntimeError('Scene is not loaded.')
		self.RAPI_rc(vrep.simxCloseScene(self.cID, vrep.simx_opmode_blocking))
		self.scene_loaded = False
	
	def start_simulation(self):
		if self.sim_running:
			raise RuntimeError('Simulation is already running.')
		self.RAPI_rc(vrep.simxSynchronous(self.cID,True))
		self.RAPI_rc(vrep.simxStartSimulation(self.cID, vrep.simx_opmode_blocking))
		self.sim_running = True
		self.enable_threaded_rendering(True)
	
	def stop_simulation(self):
		if not self.sim_running:
			raise RuntimeError('Simulation is not running.')
		self.RAPI_rc(vrep.simxStopSimulation(self.cID, vrep.simx_opmode_blocking))
		
		# Checking if the server really stopped
		while True:
			self.RAPI_rc(vrep.simxGetIntegerSignal(self.cID,'sig_debug',vrep.simx_opmode_blocking))
			e = vrep.simxGetInMessageInfo(self.cID,vrep.simx_headeroffset_server_state)
			still_running = e[1] & 1
			if not still_running:
				break
		
		self.sim_running = False
	
	def step_simulation(self):
		self.RAPI_rc(vrep.simxSynchronousTrigger(self.cID))
	
	# useful BooleanParameter's
	
	def enable_display(self, paramValue):
		self.RAPI_rc(vrep.simxSetBooleanParameter(self.cID,vrep.sim_boolparam_display_enabled,paramValue,vrep.simx_opmode_blocking))
	def enable_threaded_rendering(self, paramValue):
		self.RAPI_rc(vrep.simxSetBooleanParameter(self.cID,vrep.sim_boolparam_threaded_rendering_enabled,paramValue,vrep.simx_opmode_blocking),tolerance=vrep.simx_return_remote_error_flag)
	
	# object methods
	
	def get_object_handle(self, name):
		handle, = self.RAPI_rc(vrep.simxGetObjectHandle(self.cID, name, vrep.simx_opmode_blocking))
		return handle
	
	# "getters"
	
	def obj_get_position(self, handle, relative_to=None):
		position, = self.RAPI_rc(vrep.simxGetObjectPosition( self.cID,handle,
			-1 if relative_to is None else relative_to,
			vrep.simx_opmode_blocking))
		return position
	
	def obj_get_orientation(self, handle, relative_to=None):
		eulerAngles, = self.RAPI_rc(vrep.simxGetObjectOrientation( self.cID,handle,
			-1 if relative_to is None else relative_to,
			vrep.simx_opmode_blocking))
		return eulerAngles
	
	# (linearVel, angularVel)
	def obj_get_velocity(self, handle):
		return self.RAPI_rc(vrep.simxGetObjectVelocity( self.cID,handle,
			vrep.simx_opmode_blocking))
	
	def obj_get_joint_angle(self, handle):
		angle = self.RAPI_rc(vrep.simxGetJointPosition( self.cID,handle,
				vrep.simx_opmode_blocking
			)
		)
		return -np.rad2deg(angle[0])
	
	def obj_get_joint_force(self, handle):
		force = self.RAPI_rc(vrep.simxGetJointForce( self.cID,handle,
				vrep.simx_opmode_blocking
			)
		)
		return force
	
	def obj_read_force_sensor(self, handle):
		state, forceVector, torqueVector = self.RAPI_rc(vrep.simxReadForceSensor( self.cID,handle,
			vrep.simx_opmode_blocking))
		
		if   state & 1 != 1: # bit 0 not set
			return None # sensor data not (yet) available
		elif state & 2 == 1: # bit 1 set
			return 0 # force sensor is broken
		else:
			return forceVector, torqueVector
	
	def obj_get_vision_image(self, handle):
		resolution, image = self.RAPI_rc(vrep.simxGetVisionSensorImage( self.cID,handle,
			0, # assume RGB
			vrep.simx_opmode_blocking,
		))
		dim, im = resolution, image
		nim = np.array(im, dtype='uint8')
		nim = np.reshape(nim, (dim[1], dim[0], 3))
		nim = np.flip(nim, 0)  # horizontal flip
		#nim = np.flip(nim, 2)  # RGB -> BGR
		return nim
	
	# "setters"
	
	def obj_set_position_target(self, handle, angle):
		return self.RAPI_rc(vrep.simxSetJointTargetPosition( self.cID,handle,
			-np.deg2rad(angle),
			vrep.simx_opmode_blocking))
	
	def obj_set_velocity(self, handle, v):
		return self.RAPI_rc(vrep.simxSetJointTargetVelocity( self.cID,handle,
			v,
			vrep.simx_opmode_blocking))
	
	def obj_set_force(self, handle, f):
		return self.RAPI_rc(vrep.simxSetJointForce( self.cID,handle,
			f,
			vrep.simx_opmode_blocking))
	
	# openai/gym
	
	# Set this in SOME subclasses
	#metadata = {'render.modes': []}
	#reward_range = (-np.inf, np.inf)

	# Override in SOME subclasses
	#def _close(self): pass
	
	# Set these in ALL subclasses
	#action_space = None
	#observation_space = None
	
	# Override in ALL subclasses
	#def _step(self, action): raise NotImplementedError
	#def _reset(self): raise NotImplementedError
	#def _render(self, mode='human', close=False): return
	#def _seed(self, seed=None): return []
	
	def _close(self):
		if self.sim_running:
			self.stop_simulation()
		# Closing the scene is unnecessary
		#if self.scene_loaded:
		#	self.close_scene()
		if self.connected:
			self.disconnect()
	
