
The testing tissues, so one can inspect them.

```python
import os, sys
import torch
import mintflow
import mintflow.utils_multislice


tissue = torch.load(
    os.path.join(
        'path_to_output',
        'TestingListTissue',
        'tissue_test_1.pt'
    ),
    map_location='cpu'
)

# now `tissue` is an instance of `mintflow.utils_multislice.Slice`.
# You can access different fields like `tissue.adata` and `tissue.adata_before_scppnormalize_total`
```




