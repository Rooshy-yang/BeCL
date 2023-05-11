# Behavior Contrastive Learning (BeCL) 

This is the official codebase for ICML 2023 paper [BeCL:Behavior Contrastive Learning for Unsupervised Skill Discovery](https://arxiv.org/abs/2305.04477), which utilizes contrastive learning as intrinsic motivation for unsupervised skill discovery. 

If you find this paper useful for your research, please cite:
```
@misc{yang2023behavior,
      title={Behavior Contrastive Learning for Unsupervised Skill Discovery}, 
      author={Rushuai Yang and Chenjia Bai and Hongyi Guo and Siyuan Li and Bin Zhao and Zhen Wang and Peng Liu and Xuelong Li},
      year={2023},
      eprint={2305.04477},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

This codebase is built on top of the [Unsupervised Reinforcement Learning Benchmark (URLB) codebase](https://github.com/rll-research/url_benchmark). Our method `BeCL` is implemented in `agents/becl.py` and the config is specified in `agents/becl.yaml`.

To pre-train BeCL, run the following command:

``` sh
python pretrain.py agent=becl domain=walker seed=3
```

This script will produce several agent snapshots after training for `100k`, `500k`, `1M`, and `2M` frames and snapshots will be stored in `./models/states/<domain>/<agent>/<seed>/ `. (i.e. the snapshots path is `./models/states/walker/becl/3/ `). 

To finetune BeCL, run the following command:

```sh
python finetune.py task=walker_stand obs_type=states agent=becl reward_free=false seed=3 domain=walker snapshot_ts=2000000
```

This will load a snapshot stored in `./models/states/walker/becl/3/snapshot_2000000.pt`, initialize `DDPG` with it (both the actor and critic), and start training on `walker_stand` using the extrinsic reward of the task.

## Requirements

We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the installation ends you can activate your environment with
```sh
conda activate urlb
```

## Available Domains
We support the following domains.
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` |

### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```
The console output is also available in the form:
```
| train | F: 6000 | S: 3000 | E: 6 | L: 1000 | R: 5.5177 | FPS: 96.7586 | T: 0:00:42
```
a training entry decodes as
```
F  : total number of environment frames
S  : total number of agent steps
E  : total number of episodes
R  : episode return
FPS: training throughput (frames per second)
T  : total training time
```
