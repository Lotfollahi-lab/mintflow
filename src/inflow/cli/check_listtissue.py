
'''
Extensive checks on training and testing list tissues.
'''


import os, sys
import numpy as np
from .. import utils_multislice

def check(train_list_slice:utils_multislice.ListSlice, test_list_slice:utils_multislice.ListSlice):
    assert isinstance(train_list_slice, utils_multislice.ListSlice)
    assert isinstance(test_list_slice, utils_multislice.ListSlice)

    # if `sliceid_to_checkUnique` are the same --> they have to be totally the same
    for sl1 in train_list_slice.list_slice:
        for sl2 in test_list_slice.list_slice:
            assert (
                len(set(
                    sl1.adata.obs[sl1.dict_obskey['sliceid_to_checkUnique']]
                )) == 1
            ), "In training set one anndata contains more than one tissue, according to the fed `sliceid_to_checkUnique`={}".format(sl1.dict_obskey['sliceid_to_checkUnique'])

            assert (
                len(set(
                    sl2.adata.obs[sl2.dict_obskey['sliceid_to_checkUnique']]
                )) == 1
            ), "In testing set one anndata contains more than one tissue, according to the fed `sliceid_to_checkUnique`={}".format(sl2.dict_obskey['sliceid_to_checkUnique'])

            if set(sl1.adata.obs[sl1.dict_obskey['sliceid_to_checkUnique']]) == set(sl2.adata.obs[sl2.dict_obskey['sliceid_to_checkUnique']]):
                assert sl1 == sl2, "Two tissues in training/testing set are assigned the slice identifier {}, but they are not the same.".format(set(sl1.adata.obs[sl1.dict_obskey['sliceid_to_checkUnique']]))


    # TODO:HERE check if the testing batch IDs are a subset of training batch IDs
