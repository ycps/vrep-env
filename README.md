# vrep-env

V-REP integrated with OpenAI Gym.
This project aims to provide a superclass for V-REP gym environments.
It is analogous to [MuJoCo-env](https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py) for MuJoCo.

<table>
  <tr>
    <td><img src="/assets/screenshot.png?raw=true" width="200"></td>
    <td><img src="/assets/vrep-cartpole.png?raw=true" width="200"></td>
  </tr>
</table>

## Dependencies

- [V-REP](http://www.coppeliarobotics.com/downloads.html)
- [OpenAI Gym](https://github.com/openai/gym)

## Installation

In order to smooth the installation process, define the variables `VREP_PATH` and `VREP_SCENES_PATH` that contain your V-REP scenes.

Example:
```bash
export VREP_PATH=/example/some/path/to/V-REP_PRO_EDU_V3_4_0_Linux/
export VREP_SCENES_PATH=/example/again/V-REP_PRO_EDU_V3_4_0_Linux/scenes/
```
These variables will be used as default if the respective argument is not provided.
Next, simply install via pip:
```bash
pip3 install --upgrade git+https://github.com/ycps/vrep-env.git#egg=vrep_env
```

## Creating your own V-REP Gym environments

In order to create your own V-REP Gym environments, simply extend the [`VrepEnv`](vrep_env/vrep_env.py) class and fill in the gaps.
You may use the [`ExampleVrepEnv`](examples/envs/example_vrep_env.py) as a template base or check the fully functional [`HopperVrepEnv`](examples/envs/hopper_vrep_env.py) (similar to the [MuJoCo](https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper.py) / [Roboschool](https://github.com/openai/roboschool/blob/master/roboschool/gym_mujoco_walkers.py) Hopper)

## Usage

Before starting your environment, an instance of V-REP should already be running. It uses port 19997 by default, but it can be overriden in class initialization.
Check the [`HopperVrepEnv`](examples/envs/hopper_vrep_env.py) for a simple running example.
It can be run as:
```bash
python3 hopper_vrep_env.py
```
If everything was installed correctly, you should see a random agent struggling to hop:

<img src="/assets/hopper-random.gif?raw=true" width="400">

You may have to register the envs as following:
```python3
register(id='VrepCartPole-v0', entry_point='cartpole_vrep_env:CartPoleVrepEnv', max_episode_steps=200, reward_threshold=195.0)
register(id='VrepCartPoleContinuous-v0', entry_point='cartpole_continuous_vrep_env:CartPoleContinuousVrepEnv', max_episode_steps=200, reward_threshold=195.0)
register(id='VrepHopper-v0', entry_point='hopper_vrep_env:HopperVrepEnv', max_episode_steps=1000)
```

## Example Environments

| Environment Id | Observation Space | Action Space | tStepL | BasedOn |
|---|---|---|---|---|
|VrepCartPole-v0|Box(4)|Discrete(2)|200|[CartPole-v1](https://gym.openai.com/envs/CartPole-v1)|
|VrepCartPoleContinuous-v0|Box(4)|Box(1)|200|[CartPole-v1](https://gym.openai.com/envs/CartPole-v1)|
|VrepHopper-v0|Box(25)|Box(3)|1000|[Hopper-v*](https://github.com/openai/gym/blob/master/gym/envs/mujoco/hopper.py)|

### VrepCartPole-v0
  Based on Gym CartPole-v1 (cart-pole problem described by Barto, Sutton, and Anderson).
  An agent trained in CartPole-v1 may be able to succeed in VrepCartPole-v0 without additional training.
### VrepCartPoleContinuous-v0
  Similar to VrepCartPole-v0, but with continuous actions values.
### VrepHopper-v0
  Loosely based on MuJoCo/Roboschool/PyBullet Hopper, but the dynamics act numerically different.
  (Warning: it is not known if this env is learnable nor if the model is capable of hopping.)

## Similar projects

There are other similar projects that attempt to wrap V-Rep and create a gym interface.
Some of these projects also contain interesting scenes for learning different tasks.

- https://github.com/ctmakro/vrepper
- https://github.com/fgolemo/vrepper
- https://github.com/bhyang/gym-vrep
- https://github.com/kbys-t/gym_vrep
- https://github.com/sayantanauddy/acrobotVREP
