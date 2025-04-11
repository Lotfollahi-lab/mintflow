<img src="https://github.com/Lotfollahi-lab/mintflow/blob/main/docs/_static/mintflow_logo_readme.png" width="800" alt="mintflow-logo">

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Lotfollahi-lab/mintflow/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/Lotfollahi-lab/mintflow?logo=GitHub&color=yellow)](https://github.com/Lotfollahi-lab/mintflow/stargazers)
[![PyPI](https://img.shields.io/pypi/v/mintflow.svg)](https://pypi.org/project/mintflow)
[![PyPIDownloads](https://static.pepy.tech/badge/mintflow)](https://pepy.tech/project/mintflow)
[![Docs](https://readthedocs.org/projects/mintflow/badge/?version=latest)](https://mintflow.readthedocs.io/en/stable/?badge=stable)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

MintFlow (**M**icroenvironment-induced and **IN**trinsic **T**ranscriptomic **FLOW**s) is a package to decompose spatial transcriptomics data into microenvironment-induced and intrinsic gene expression components. It interoperates with the scverse ecosystem [scverse](https://scverse.org/) to enable seamless analysis workflows of spatial transcriptomics data to identify spatial biomarkers. 

## Installing the Python Environment
 **SANGER INTERNAL**: The environment is already available on farm.

To activate it:
```commandline
module load cellgen/conda
conda activate /nfs/team361/aa36/PythonEnvs_2/envinflowdec27/
```

Alternatively, you can create the python environment yourself:
```commandline
git clone https://github.com/Lotfollahi-lab/mintflow.git  # clone the repo
cd ./mintflow/
conda env create -f environment.yml --prefix SOME_EMPTY_PATH
```

## Installing WandB
It's highly recommended to setup wandb before proceeding.

To do so:
- Go to https://wandb.ai/ and create an account.
- Create a project called "MintFlow".

## Quick Start
You can use mintflow as a local package, because it's not pip installable at the moment.

To do so:
```commandline
git clone https://github.com/Lotfollahi-lab/mintflow.git  # clone the repo
cd ./mintflow/
```
The easiest way to run MintFlow is through the command line interface (CLI).
This involves two steps
1. Creating four config files (you duplicate/modify template config files).
2. Running mintflow with a single command line.

### Rule of thumbs §1 for modifying the config files
In the template config files, there are `TODO`-s of different types that you may need to modify
- Category 1: `TODO:ESSENTIAL:TUNE`: the basic/essential parts to run mintflow.
- Category 2: `TODO:TUNE`: less essneitial and/or technical details.
- Category 3: `TODO:check`: parameters of even less importance compared to category 1 and category 2.

If you are, for example, a biologist with no interest/experience in computational methods, you can only modify "Category 1" above and leave the rest of configurations untouched.
"Category 2" and "Category 3" come next in both priority and the level of details.

### Step 1 of Using the CLI: Making 4 config files
Please follow these steps
- Training data config file:
    - Make a copy of `./cli/SampleConfigFiles/config_data_train.yml` and rename it to `YOUR_CONFIG_DATA_TRAIN.yml`
    - Read the block of comments tarting with *"# MintFlow expects a list of .h5ad files stored on disk, ..."*.
    - Modify some parts marked by `TODO:...` and according to *"Rule of thumbs §1"* explained above.


- Testing data config file:
    - Make a copy of `YOUR_CONFIG_DATA_TRAIN.yml` and rename it to `YOUR_CONFIG_DATA_TEST.yml`
    - Rename all ocrrences of `config_dataloader_train` to `config_dataloader_test`


- Model config file:
    - Make a copy of `./cli/SampleConfigFiles/config_model.yml` and rename it to `YOUR_CONFIG_MODEL.yml`.
    - Modify some parts marked by `TODO:...` and according to *"Rule of thumbs §1"* explained above.


- Training config file:
    - Make a copy of `./cli/SampleConfigFiles/config_training.yml` and rename it to `YOUR_CONFIG_TRAINING.yml`.
    - Modify some parts marked by `TODO:...` and according to *"Rule of thumbs §1"* explained above.

### Step 2 of Using the CLI: Running MintFlow

```commandline
cd ./mintflow/  # if you haven't already done it above.
cd ./cli/

python mintflow_cli.py \
--file_config_data_train YOUR_CONFIG_DATA_TRAIN.yml \
--file_config_data_test YOUR_CONFIG_DATA_TEST.yml \
--file_config_model YOUR_CONFIG_MODEL.yml \
--file_config_training YOUR_CONFIG_TRAINING.yml \
--path_output "./Your/Output/Path/ToDump/Results/" \
--flag_verbose "True" \
```
The recommended way of accessing MintFlow predictions is by `adata_mintflowOutput_norm.h5ad` and `adata_mintflowOutput_unnorm.h5ad` created in the provided `--path_output`and `adata.obsm` and `adata.uns` in these files.
In the former file `..._norm.h5ad` the readcount matrix `adata.X` as well as MintFlow predictions Xint and Xspl are row normalised, while in the latter file `_unnorm.h5ad` they are not.

MintFlow dumps a README file in the provided `--path_output`, as well as each subfolder therein.

## Common Issues
- Use absolute paths (and not relative paths like `../../some/path/`) in the config files, as well as when running `python mintflow_cli.py ...`.
- TODO: intro to the script for tune window width.
- It's common to face out of memory issue in the very last step where the big anndata objects `adata_mintflowOutput_norm.h5ad` and `adata_mintflowOutput_unnorm.h5ad` are created and dumped.
If that step fails, the results are still accesible in the output path the subfolder `CheckpointAndPredictions/`.
One can laod the `.pt` files by
```python
import torch
dict_results = torch.load(
    "the/output/path/CheckpointAndPredictions/predictions_slice_1.pt",
    map_location='cpu'
)
```

## Release notes
TODOTODO
See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/sebastianbirk/celldino/issues
[changelog]: https://celldino.readthedocs.io/latest/changelog.html
[link-docs]: https://celldino.readthedocs.io
[link-api]: https://celldino.readthedocs.io/latest/api.html
