# Wall-Following-Robot-RL

## Dependencies
Along with python version >= 3, to ensure the code runs similar to ours, the requirements in requirements.txt must be installed.

To install the requirements run:

```terminal
pip install -r riquirements.txt
```

It is to be noted that installing gymnasium[box2d] has a known issue where building the wheels fails. This issue is resolved differently depending on your machine type.

On our machines the following resolved

#### Mac
install swig before installing gymnasium[box2d]
```terminal
pip install swig
```

#### Windows
The instructions in the following video were helpful:

https://youtu.be/gMgj4pSHLww?si=asoE1KxlGiYnBwop


## Training

To train the model as done in the report, the vectorizedMain.py needs to be run as defined below.


#### Rendering

The code can be run with or without rendering the training process. Although not rendering may quicken the process, however, as our code is well optimized to not take long anyways, we recomend rendering to see the model learn. 

By default, vectorizedMian.py is set up to render.

If you choose not to render, run the file with the following argument:

```terminal
python vectorizedMain.py --render_mode=0
```
##

#### Training for fixed or varying constant speed across episodes

By default, the vectorizedMain.py file is set up to run with a fixed constant speed for each episode.

If you wan to train for variable speed per episode, run the file with the following argument

```terminal
python vectorizedMain.py --var_speed=1
```
##

#### Training for accleration and brake along with steering

By default, the vectorizedMain.py file is set up to run without  accleration and brake actions and only steering.

If you wan to train for accleration and brake as well, run the file with the following argument

```terminal
python vectorizedMain.py --accel_brake=1
```

##

#### Train the model without penalizing for oscilations

If you want to try running the model simplifying the reward function to just be the shifted negative squared CTE, run vectorized with the following command:

```terminal
python vectorizedMain.py --penalize_oscl=0
```

##

#### Loading a pre-trained model or policy

First note that in this context, policies are the trained models that correspond to the data stats before each training iterations. These are stored in the './policies' directory.

Models are the evaluated models stored at each evaluation during training, and also after the final evaluation after training is terminated.

To load a pretrained model or policy run either of the following as required:

example model: 'TD3_010'
```terminal
python vectorizedMain.py --load_model=010
```
example policy: 'policy_12'
```terminal
python vectorizedMain.py --load_policy=12
```


## Testing

Similar to loading a model or a policy as before, run the following as required:

example model: 'TD3_010'
```terminal
python test.py --load_model=010
```
example policy: 'policy_12'
```terminal
python test.py --load_policy=12
```

#

Consider you have just trained the model and simply want to test the final trained model:

```terminal
python test.py --load_model=Final
```

## Evaluating with Benchmarks

For this, simply run the following to get the all the matplot lib plots and evaluations printed in the console.

```terminal
python benchmarks.py
```