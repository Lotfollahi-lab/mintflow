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

## Step 1: Installation via PyPi

Install MintFlow via pip:
```
pip install mintflow
```

Or install including optional dependencies required for running tutorials with:
```
pip install mintflow[all]
```

## Step 2: Additional Libraries

To use MintFlow, you need to install some additional external libraries. These include:
- [PyTorch Scatter]
- [PyTorch Sparse]

To install these libraries, after installing MintFlow run:

```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions, respectively.

For example, for PyTorch 2.6.0 and CUDA 12.4, type:
```
pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

[Mambaforge]: https://github.com/conda-forge/miniforge
[Docker]: https://www.docker.com
[PyTorch]: http://pytorch.org
[PyTorch Scatter]: https://github.com/rusty1s/pytorch_scatter
[PyTorch Sparse]: https://github.com/rusty1s/pytorch_sparse
