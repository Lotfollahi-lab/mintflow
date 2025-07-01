# Installation

MintFlow is available for Python 3.10 and 3.11.

We do not recommend installation on your system Python. Please set up a virtual
environment, e.g. via venv or conda through the [Mambaforge] distribution, or
create a [Docker] image.

To set up and activate a virtual environment with venv, run:

```
python3 -m venv ~/.venvs/mintflow
source ~/.venvs/mintflow/bin/activate
```

To create and activate a conda environment instead, run:

```
conda create -n mintflow python=3.11
conda activate mintflow
```
## Step 1: Install PyTorch
Visit [Pytorch website] and install its appropriate version based on your OS and compute platform.

## Step 2: Install PyTorch Geometric
### Step 2.1: Figure out your pytorch and cuda versions
To learn your pytorch version, if you installed pytorch with conda, you can run
```commandline
conda list | grep torch
```
or if you installed pytorch via pip, you can run
```commandline
pip list | grep torch
```
and the version number is printed next to "torch".

If you want to use GPU acceleration, run the following command to know about your CUDA version:
```commandline
nvidia-smi
```

### Step 2.2: Install additional libraries related to PyTorch Geometric
Before installing PyTorch geometric, you need to install some additional external libraries. These include:
- [PyG-lib]
- [PyTorch Scatter]
- [PyTorch Sparse]

To install these libraries, run
```
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions, respectively.

For example, for PyTorch 2.6.0 and CUDA 12.4, type:
```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```
If you have chosen not to use GPU acceleration, `${CUDA}` should be replaced by "cpu".

### Step 2.3: Install PyTorch Geometric
Run
```commandline
pip install torch_geometric
```

## Step 3: Install MintFlow

Install MintFlow via pip:
```
pip install mintflow
```

Or install including optional dependencies required for running tutorials with:
```
pip install mintflow[all]
```



[Mambaforge]: https://github.com/conda-forge/miniforge
[Docker]: https://www.docker.com
[PyTorch]: http://pytorch.org
[PyTorch website]: http://pytorch.org
[PyTorch Scatter]: https://github.com/rusty1s/pytorch_scatter
[PyTorch Sparse]: https://github.com/rusty1s/pytorch_sparse
[PyTorch geometric website]: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
[PyG-lib]: https://pyg-lib.readthedocs.io/en/latest/
