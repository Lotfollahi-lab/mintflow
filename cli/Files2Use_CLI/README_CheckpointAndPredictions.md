
It's recommended to read MintFlow outputs from `adata_mintflowOutput_norm.h5ad` and `adata_mintflowOutput_unnorm.h5ad` in the output path.
If you choose not to, this folder contains one `.pt` file per testing tissue sample.

All `.pt` files can be loaded by `torch.load`.


This folder contains
- `mintflow_model.pt`: the MintFlow trained checkpoint.
- `predictions_slice_X.pt` where `X` varies between one and the number of testing samples. It contains a dictionary with many keys related to MintFlow predictions (`muxint`, `mu_z`, `mu_sin`).





