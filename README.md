## Installation

```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
$ mamba env create -f conda_environment.yaml
```

## Path determination

1. dataset_path in config/task
2. task in config(workspace)

## training

```shell
$ conda activate robodiff
$ wandb login
$ python train.py --config-name=train_diffusion_transformer_hybrid_workspace.yaml
```