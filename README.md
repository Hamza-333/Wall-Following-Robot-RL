# Reinforcement Learning for Robot Path-Following 

## Dependencies
Along with Python version `>= 3`, to ensure the code runs similarly to ours, the requirements in requirements.txt must be installed.

To install the requirements run:

```terminal
pip install -r requirements.txt
```

It is to be noted that installing gymnasium[box2d] has a known issue where building the wheels fails. This issue is resolved differently depending on your machine type.

On our machines, the following resolved

#### Mac
install swig before installing gymnasium[box2d]
```terminal
pip install swig
```

#### Windows
The instructions in the following video were helpful:

https://youtu.be/gMgj4pSHLww?si=asoE1KxlGiYnBwop

## Terminolgy

First note that our implementation the following convention for naming policies was set:

- Policies, stored as "Policy_", are the trained models that correspond to the data stats before each training iteration. These are stored in the './policies' directory.

- Models, stored as "TD3_",  are the evaluated models stored at each evaluation during training and after the final evaluation after training is terminated. They are stored in the directory './pytorch_models'.


## Training

To train the model as done in the report for our main task, constant speed path following, the vectorizedMain.py needs to be run without any arguments passed to it.

For training the model for the extended tasks, variable constant speed/acceleration, and other settings, the vectorizedMain.py needs to be run as defined below.


#### Rendering

The code can be run with or without rendering the environment simulation visually for the training process. Although not rendering may quicken the process, however, as our code is well optimized to not take long anyway, we recommend rendering to see the model learn. 

By default, vectorizedMian.py is set up to render.

If you choose not to render, run the file with the following argument:

```terminal
python vectorizedMain.py --render_mode=0
```
##

#### Training for fixed or varying constant speed across episodes

By default, the vectorizedMain.py file is set up to run with a fixed constant speed for each episode.

If you want to train for variable speed per episode, run the file with the following argument

```terminal
python vectorizedMain.py --var_speed=1
```
##

#### Training for acceleration (gas and brake) along with steering

By default, the vectorizedMain.py file is set up to run without acceleration (gas and brake) actions and only with the steering action.

If you want to train for acceleration (gas and brake) actions as well, run the file with the following argument

```terminal
python vectorizedMain.py --accel_brake=1
```

##

#### Train the model without penalizing for oscillations

If you want to try running the model simplifying the reward function to just be the shifted negative squared CTE, run vectorized with the following command:

```terminal
python vectorizedMain.py --penalize_oscl=0
```

##

#### Loading a pre-trained model or policy before training(functionality for curriculum learning)

In our current implementation, please note that this functionality is only supported for the main task: constant speed.

To load a pre-trained model or policy run either of the following as required:

example model: 'TD3_Final'
```terminal
python vectorizedMain.py --load_model=Final
```
example policy: 'policy_14'
```terminal
python vectorizedMain.py --load_policy=14
```

## Testing

# Our best results
For our best models for the main task, run the following:

Constant speed:
```terminal
python test.py --load_policy=best1
```

```terminal
python test.py --load_policy=best2
```

Similarly, for extended tasks, passing these arguments is enough

Variable speed:
```terminal
python test.py --load_policy=best1 --var_speed=1
python test.py --load_policy=best2 --var_speed=1
```
Acceleration:
```terminal
python test.py --load_policy=best1 --accel_brake=1
python test.py --load_policy=best2 --accel_brake=1
```

# Testing your own trained polices
Similar to loading a model or a policy as before, run the following as required for our main task:

example model: 'TD3_010'
```terminal
python test.py --load_model=010
```
example policy: 'policy_14'
```terminal
python test.py --load_policy=14
```
#

Consider you have just trained the model and simply want to test the final trained model:

```terminal
python test.py --load_model=Final
```

If you want to test for the extended tasks, pass the following arguments along as required:

For variable speed:
```terminal
--var_speed=1
```

For acceleration:
```terminal
--accel_brake=1
```

example usage for variable speed model: 'TD3_VAR_0'
```terminal
python test.py --load_model=0 --var_speed=1
```
example usage for acceleration policy: 'Policy_14'
```terminal
python test.py --load_policy=14 --accel_brake=1
```

Our best results are policies are as following:

```terminal
python test.py --load_policy= --accel_brake=1
```

```terminal
python test.py --load_policy=14 --accel_brake=1
```



## Evaluating with Benchmarks

For this, simply run the following to get all the matplotlib plots and evaluations printed in the console. 

```terminal
python benchmarks.py
```

## Acknowledgements
- University of Toronto's CSC2626 Assignment 1 repository: https://github.com/florianshkurti/csc2626w22/tree/master/assignments/A1
