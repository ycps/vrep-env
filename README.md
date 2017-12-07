# vrep-env

V-REP integrated with OpenAI Gym.
This project aims to provide a superclass for V-REP gym environments.
It is analogous to [MuJoCo-env](https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py) for MuJoCo.

<img src="/assets/screenshot.png?raw=true">

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


## Similar projects

There are other similar projects that attempt to wrap V-Rep and create a gym interface.
Some of these projects also contain interesting scenes for learning different tasks.

- https://github.com/ctmakro/vrepper
- https://github.com/fgolemo/vrepper
- https://github.com/bhyang/gym-vrep
- https://github.com/kbys-t/gym_vrep
- https://github.com/sayantanauddy/acrobotVREP
