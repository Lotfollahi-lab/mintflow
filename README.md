<img src="https://github.com/Lotfollahi-lab/mintflow/blob/main/docs/_static/mintflow_logo_readme.png" width="400" alt="mintflow-logo">

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://github.com/Lotfollahi-lab/mintflow/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/Lotfollahi-lab/mintflow?logo=GitHub&color=yellow)](https://github.com/Lotfollahi-lab/mintflow/stargazers)
[![PyPI](https://img.shields.io/pypi/v/mintflow.svg)](https://pypi.org/project/mintflow)
[![PyPIDownloads](https://static.pepy.tech/badge/mintflow)](https://pepy.tech/project/mintflow)
[![Docs](https://readthedocs.org/projects/mintflow/badge/?version=latest)](https://mintflow.readthedocs.io/en/stable/?badge=stable)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

MintFlow (**M**icroenvironment-induced and **IN**trinsic **T**ranscriptomic **FLOW**s) is a package to decompose spatial transcriptomics data into microenvironment-induced and intrinsic gene expression components. It interoperates with the [scverse](https://scverse.org/) ecosystem to enable seamless analysis workflows of spatial transcriptomics data to identify spatial biomarkers.

## Installation

```commandline
pip install mintflow
```

## Installing WandB
It's highly recommended to setup wandb when using MintFlow.

To do so:
- Go to https://wandb.ai/ and create an account.
- Create a project called "MintFlow".


## Tutorials
The repository and documentation are being developed (we haven't yet released a usable pypi pacakge).
In the meanwhile, some sample notebooks are available in `docs/tutorials/notebooks/`.

## Release notes
See the [changelog][changelog].

## Contact

## Resources
- An installation guide, tutorials and API documentation is available in the [documentation](https://mintflow.readthedocs.io/).
- Please use [issues](https://github.com/Lotfollahi-lab/mintflow/issues) to submit bug reports.
- If you would like to contribute, check out the [contributing guide](https://mintflow.readthedocs.io/en/latest/contributing.html).
- If you find MintFlow useful for your research, please consider citing the MintFlow manuscript.

## Reference
```
@article{Akbarnejad2025,
  author    = {Akbarnejad, A. et al.},
  title     = {Mapping and reprogramming microenvironment-induced cell states in human disease using generative AI},
  journal   = {bioRxiv},
  year      = {2025},
  doi       = {10.1101/2025.06.24.661094},
  url       = {https://doi.org/10.1101/2025.06.24.661094}
}
```

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/sebastianbirk/celldino/issues
[changelog]: https://celldino.readthedocs.io/latest/changelog.html
[link-docs]: https://celldino.readthedocs.io
[link-api]: https://celldino.readthedocs.io/latest/api.html
