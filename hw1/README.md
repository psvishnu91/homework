# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, MuJoCo version 1.31, OpenAI Gym

**Note**: MuJoCo versions until 1.5 do not support NVMe disks therefore won't be compatible with recent Mac machines.
There is a request for OpenAI to support it that can be followed [here](https://github.com/openai/gym/issues/638).

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* Ant-v1.pkl
* HalfCheetah-v1.pkl
* Hopper-v1.pkl
* Humanoid-v1.pkl
* Reacher-v1.pkl
* Walker2d-v1.pkl

The name of the pickle file corresponds to the name of the gym environment.

------
# Executing the code

## Section 3: Behavior cloning
This involves sampling (state->action) pairs from an expert policy and
learning a supervised model for various tasks.

### Code to sample an expert
``` bash
$ python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1  \
	humanoid_rollouts_20.pkl \
	--render \
	--num_rollouts 20
```

### Code to learn a supervised model to clone behavior
``` bash
$ python bclone.py  \
	--rollout humanoid_rollouts_50.pkl \
	--output humanoid_bc_3l_fc_h100_500e.pkl \
	--epochs 500
```

### Code to simulate and test the cloned behavior
``` bash
$ python run_expert.py \
	humanoid_bc_3l_fc_h100_500e.pkl  \
	Humanoid-v1  \
	--user_policy \
	--render \
	--num_rollouts 10
```
### Code to train the dagger agent. This will also save videos of the agent after every dagger iter.
``` bash
python dagger.py \
	--expert experts/Humanoid-v1.pkl \
	--init-rollout-sz 20 \
	--init-epochs 200 \
	--rollout-sz 20 \
	--epochs 200 \
	--max-dagger-iters 10 \
	--output-path humanoid_dagger.pkl \
	--env Humanoid-v1 \
	--capture-dir videos_dagger/
```
#### Video
[![Watch the video](https://raw.githubusercontent.com/psvishnu91/homework/master/hw1/resources/dagger-humanoid.png ==250x250)](https://www.youtube.com/watch?v=pdywstcEdaw)
#### Humanoid dagger training
![Humanoid dagger training](https://raw.githubusercontent.com/psvishnu91/homework/master/hw1/resources/dagger-humanoid-training.png)
