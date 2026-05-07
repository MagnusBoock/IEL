# Isomorphic Embedding Learning

## Overview
This codebase contains the official implementation of Isomorphic Embedding Learning (IEL) for the paper "Hitting Time Isomorphism for Multi-Stage Planning with Foundation Policies".
The implementation is based on [Hilbert Foundation Policies](https://seohong.me/projects/hilp/).


## Examples
```
# IEL on antmaze-large-diverse
python main.py --run_group EXP --agent_name iel --algo_name iel --seed 0 --env_name antmaze-large-diverse-v2

# IEL on kitchen-partial
python main.py --run_group EXP --agent_name iel --algo_name iel --seed 0 --env_name kitchen-partial-v0 --train_steps 500000 --eval_interval 50000 --save_interval 500000 
```

## Installation
```
conda env create -f env.yaml
```

### Install MuJoCo
if mujoco is not already installed, you can install it by running the following commands in your terminal:
```
cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
rm -r mujoco210-linux-x86_64.tar.gz 
```

### Set up environment variables

Add the following lines to `~/.bashrc`: 

```
# 1. Prioritize NVIDIA drivers (Crucial for EGL hardware acceleration)
export LD_LIBRARY_PATH=/usr/lib/nvidia:$LD_LIBRARY_PATH
 
# 2. Legacy MuJoCo path (keep it if other components need it, but it's secondary now)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<USER>/.mujoco/mujoco210/bin
 
# 3. Force BOTH MuJoCo and PyOpenGL to use EGL for headless work
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
 
# 4. Tell NVIDIA where the EGL vendor config is (Often fixes the <exception str() failed> error)
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json 
```

### Wandb Credentials
Use `wandb.login` pasting your wandb API key when prompted.



## License

MIT