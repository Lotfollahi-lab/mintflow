# MintFlow

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/sebastianbirk/mintflow/test.yaml?branch=main
[link-tests]: https://github.com/sebastianbirk/mintflow/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/mintflow

Microenvironment-induced and INtrinsic Transcriptomic FLOWs

## Installing the Python Environment
 **SANGER INTERNAL**: The environment is already available on farm.

To activate it:
```commandline
module load cellgen/conda
conda activate /nfs/team361/aa36/PythonEnvs_2/envinflowdec27/
```

Alternatively, you can create the python environment yourself:
```commandline
git clone https://github.com/Lotfollahi-lab/inflow.git  # clone the repo
cd ./inflow/
conda env create -f environment.yml --prefix SOME_EMPTY_PATH
```

## Installing WandB
It's highly recommended to setup wandb before proceeding.

To do so:
- Go to https://wandb.ai/ and create an account.
- Create a project called "inFlow".

## Quick Start
You can use inflow as a local package, because it's not pip installable at the moment.

To do so:
```commandline
git clone https://github.com/Lotfollahi-lab/inflow.git  # clone the repo
cd ./inflow/
```
The easiest way to run inflow is through the command line interface (CLI).
This involves two steps
1. Creating four config files (you duplicate/modify template config files).
2. Running inflow with a single command line.

### Rule of thumbs §1 for modifying the config files
In the template config files, there are `TODO`-s of different types that you may need to modify
- Category 1: `TODO:ESSENTIAL:TUNE`: the basic/essential parts to run inflow.
- Category 2: `TODO:TUNE`: less essneitial and/or technical details.
- Category 3: `TODO:check`: parameters of even less importance compared to category 1 and category 2.

If you are, for example, a biologist with no interest/experience in computational methods, you can only modify "Category 1" above and leave the rest of configurations untouched.
"Category 2" and "Category 3" come next in both priority and the level of details.

### Step 1 of Using the CLI: Making 4 config files
Please follow these steps
- Training data config file:
    - Make a copy of `./cli/SampleConfigFiles/config_data_train.yml` and rename it to `YOUR_CONFIG_DATA_TRAIN.yml`
    - Read the block of comments tarting with *"# Inflow expects a list of .h5ad files stored on disk, ..."*.
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

### Step 2 of Using the CLI: Running inflow

```commandline
cd ./inflow/  # if you haven't already done it above.
cd ./cli/

python inflow_cli.py \
--file_config_data_train YOUR_CONFIG_DATA_TRAIN.yml \
--file_config_data_test YOUR_CONFIG_DATA_TEST.yml \
--file_config_model YOUR_CONFIG_MODEL.yml \
--file_config_training YOUR_CONFIG_TRAINING.yml \
--path_output "./Your/Output/Path/ToDump/Results/" \
--flag_verbose "True" \
```
The recommended way of accessing inflow predictions is by `adata_inflowOutput_norm.h5ad` and `adata_inflowOutput_unnorm.h5ad` created in the provided `--path_output`and `adata.obsm` and `adata.uns` in these files.
In the former file `..._norm.h5ad` the readcount matrix `adata.X` as well as inflow predictions Xint and Xspl are row normalised, while in the latter file `_unnorm.h5ad` they are not.

Inflow dumps a README file in the provided `--path_output`, as well as each subfolder therein.

## Common Issues
- Use absolute paths (and not relative paths like `../../some/path/`) in the config files, as well as when running `python inflow_cli.py ...`.
- TODO: intro to the script for tune window width.
- It's common to face out of memory issue in the very last step where the big anndata objects `adata_inflowOutput_norm.h5ad` and `adata_inflowOutput_unnorm.h5ad` are created and dumped.
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
