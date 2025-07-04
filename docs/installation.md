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
Visit the [PyTorch website] and install its appropriate version based on your OS and compute platform.

## Step 2: Install PyTorch Geometric Dependencies
### Step 2.1: Figure out your PyTorch and CUDA versions
To learn your PyTorch version, if you installed PyTorch with conda, you can run
```commandline
conda list | grep torch
```
or if you installed PyTorch via pip, you can run
```commandline
pip list | grep torch
```

If you want to use GPU acceleration, please run the following command to know your CUDA version:
```commandline
nvidia-smi
```

### Step 2.2: Install additional libraries related to PyTorch Geometric
Before installing PyTorch Geometric, you need to install some additional external libraries. These include:
- [PyTorch Scatter]
- [PyTorch Sparse]

To install these libraries, run
```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions (see previous section how to obtain them).

For example, for PyTorch 2.6.0 and CUDA 12.4, type:
```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```
If you have chosen not to use GPU acceleration, `${CUDA}` should be replaced by "cpu".

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
