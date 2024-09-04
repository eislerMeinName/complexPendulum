# complexPendulum
ComplexPendulum is a detailed gym environment of the inverted pendulum RL environment. 
It aims at modelling real-world pendulum systems as accurate as possible.

> ### Currently under construction...
> - why is K* of LQR different to matlab lqr?
> - find trainings and evaluation setups

## Installation
This repo was written using Python3.10 with conda on Arch Linux 
and Ubuntu 20.04. Compatibilities with other OS should be feasible 
out of the box, but without guarantee.

### Conda
First install conda. 
For Ubuntu 20.04 read 
[here](https://linuxize.com/post/how-to-install-anaconda-on-ubuntu-20-04/) for Information and Installation Instructions. 
For Arch Linux read [here](https://docs.anaconda.com/anaconda/install/linux/).

### complex-Pendulum
```bash
$ conda create -n complexPendulum python=3.10
$ conda activate complexPendulum
$ git clone https://github.com/eislerMeinName/complexPendulum.git
$ cd complexPendulum/
$ pip3 install --upgrade pip
$ pip3 install -e .
```
Then test installation via:
```bash
$ python3 test.py
```
Now you should see a gui with a Pendulum swinging up to the unstable equilibrium.
You can easily evaluate this step response via:
```bash
$ python3 evaltest.py
```

## Environment
The ComplexPendulum environment is multimodal, and uses a logger class.
An example of use can be found in test.py.

```python
>>> import gymnasium as gym
>>> env = gym.make('complexPendulum-v0')
```
Overview of Parameters:
```python
>>> env = ComplexPendulum(
>>>     frequency: float,                               #control frequency
>>>     episode_len: float,                             #episode length
>>>     path: str,                                      #path to parameter file
>>>     Q: np.ndarray,                                  #Q matrix
>>>     R: np.ndarray,                                  #R matrix
>>>     gui: bool = False,                              #use GUI
>>>     actiontype: ActionType = ActionType.DIRECT,     #Defines ActionSpace
>>>     rewardtype: RewardType = RewardType.LQ,         #Defines RewardFunction
>>>     s0: np.array = None,                            #starting state
>>>     friction: bool = True,                          #use static friction
>>>     log: bool = True)                               #log step response
```
### XML-Parameter
If you want to specify different kinematic parameters of a pendulum, edit params.xml file
or write a new XML file as input to the environment.

| parameter |                                  explanation                                   | standard value |
|----------:|:------------------------------------------------------------------------------:|:--------------:|
|        mp |                             Mass of pendulum [kg]                              |      0.09      |
|         l | Distance between the mounting point and the center of mass of the pendulum [m] |     0.2376     |
|         J |                           Moment of inertia  [kg m2]                           |   7.3404e-3    |
|         m |                                Total mass [kg]                                 |     0.8633     |
|        fp |               Friction coefficient for the pendulum [N m s/rad]                |   2.5014e-4    |
|        fc |                   Friction coefficient for the cart [N s/m]                    |      0.5       |
|         g |                       Gravitational acceleration [m/s2]                        |      9.81      |
|         M |                          PWM to Force coefficient [N]                          |  11.70903084   |
|        Fs |                        Static Friction coefficient [N]                         |  0.292725771   |
|    xquant |                          Quantization of encoder [m]                           |   5.7373e-5    |
| thetaquant|                       Quantization of angle resolution                         |   0.001534     |

### ActionTypes
The ActionType models the action of the agent.
- DIRECT: directly apply pwm (dim 1)
- GAIN: apply proportional gain (dim 4)

### RewardTypes
The RewardTypes specify the reward function of the environment.
- LQ models a linear quadratic reward function.
- EXP models an exponential reward function.

### Logger
The logger logs the current step response and may plot it using matplotlib.
You can also write the data to a csv file.

<img src="readme_images/loggerexample.png">

## Agents
Besides learning an agent via RL, there are a couple of classes containing agents based on classical control engineering.

|    agent     |                          explanation                           | actiontype |
|:------------:|:--------------------------------------------------------------:|:----------:|
| proportional |           General interface for proportional agent.            |    GAIN    |
|      lq      |            Agent based on linear quadratic control.            |    GAIN    |
|   swing-up   | Agent used to swing up the pendulum by pumping energy into it. |   DIRECT   |
|   combined   |       Combines a swing-up agent and a proportional agent       |   DIRECT   |

## Evaluator
The evaluator class evaluates a logged step response on different response chunks,
on undiscounted return evaluation-setups and classic step control loop performance requirements.

### EvaluationDataType
EvaluationDataType is an enum showing which part of the data should be evaluated.
- COMPLETE: The complete step response including swing-up.
- SWING_UP: Only the swing-up.
- STEP: Without swing-up.

### EvalSetup
Describes one evaluation setup: reward function (setupType), name; Q, R, k if needed.

## Learn Script
Learn an agent with a classic RL algorithm and a wide range of possible environment parameters.

## Log Script
Load a learned agent and visualize/ log a step response.

## Evaluation Script
Evaluate a learned agent using the evaluator class on a wide range of reward functions and classic control theory evaluation criteria.

> ## Citation
> Nothing to cite, yet...
