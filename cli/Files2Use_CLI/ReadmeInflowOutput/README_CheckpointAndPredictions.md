
It's recommended to read inflow outputs from `adata_inflowOutput_norm.h5ad` and `adata_inflowOutput_unnorm.h5ad` in the output path.
If you choose not to, this folder contains one `.pt` file per testing tissue sample.

All `.pt` files can be loaded by `torch.load`.


This folder contains
- `inflow_model.pt`: the inflow trained checkpoint.
- `predictions_slice_X.pt` where `X` varies between one and the number of testing samples. It contains a dictionary with many keys related to inflow predictions (`muxint`, `mu_z`, `mu_sin`).





