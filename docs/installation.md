# Installation

Inigen is available for Python 3.10 and 3.11.

We do not recommend installation on your system Python. Please set up a virtual
environment, e.g. via venv or conda through the [Mambaforge] distribution, or
create a [Docker] image.

To set up and activate a virtual environment with venv, run:

```
python3 -m venv ~/.venvs/inigen
source ~/.venvs/inigen/bin/activate
```

To create and activate a conda environment instead, run:

```
conda create -n inigen python=3.11
conda activate inigen
```

## Step 1: Installation via PyPi

Install Inigen via pip:
```
pip install inigen
```

Or install including optional dependencies required for running tutorials with:
```
pip install inigen[all]
```

## Step 2: Additional Libraries

To use Inigen, you need to install some additional external libraries. These include:
- [PyTorch Scatter]
- [PyTorch Sparse]

To install these libraries, after installing inigen run:

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
