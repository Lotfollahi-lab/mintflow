
'''
Extensive checks on training and testing list tissues.
'''


import os, sys
import numpy as np
from .. import utils_multislice
from pprint import pprint

def check(train_list_slice:utils_multislice.ListSlice, test_list_slice:utils_multislice.ListSlice):
    assert isinstance(train_list_slice, utils_multislice.ListSlice)
    assert isinstance(test_list_slice, utils_multislice.ListSlice)

    # if `sliceid_to_checkUnique`s are the same --> they have to be totally the same
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
                flag_sl1_eq_sl2, msg_unequalpart = sl1.custom_eq_with_namemismatch(sl2)
                if not flag_sl1_eq_sl2:
                    raise Exception(
                        "Two tissues in training/testing set are assigned the slice identifier {}, but they are not the same, according to their {} .".format(
                            set(sl1.adata.obs[sl1.dict_obskey['sliceid_to_checkUnique']]),
                            msg_unequalpart
                        )
                    )
                assert sl1 == sl2, "Two tissues in training/testing set are assigned the slice identifier {}, but they are not the same according to their {}".format(
                    set(sl1.adata.obs[sl1.dict_obskey['sliceid_to_checkUnique']]),
                    msg_unequalpart
                )


    # check if the testing batch IDs are a subset of training batch IDs
    set_bid_train = set([
        sl._get_batchid() for sl in train_list_slice
    ])
    set_bid_test = set([
        sl._get_batchid() for sl in test_list_slice
    ])

    if not set_bid_test.issubset(set_bid_train):
        raise Exception(
            "The following biobatch identifiers are present in the test set, while they haven't been present in the training set. This is prohibited. \n {}".format(
                set_bid_test.difference(set_bid_train)
            )
        )



    # check if the mapping dicts in two list slice-s are the same.
    if train_list_slice.map_CT_to_inflowCT != test_list_slice.map_CT_to_inflowCT:
        print("train_list_slice.map_CT_to_inflowCT:")
        pprint(train_list_slice.map_CT_to_inflowCT)

        print("test_list_slice.map_CT_to_inflowCT:")
        pprint(test_list_slice.map_CT_to_inflowCT)

        raise Exception(
            "Training list tissue and testing list tissue do not have the same `map_CT_to_inflowCT`. See above for the content of each."
        )

    if train_list_slice.map_Batchname_to_inflowBatchID != test_list_slice.map_Batchname_to_inflowBatchID:
        print("train_list_slice.map_Batchname_to_inflowBatchID:")
        pprint(train_list_slice.map_Batchname_to_inflowBatchID)

        print("test_list_slice.map_Batchname_to_inflowBatchID:")
        pprint(test_list_slice.map_Batchname_to_inflowBatchID)

        raise Exception(
            "Training list tissue and testing list tissue do not have the same `map_Batchname_to_inflowBatchID`. See above for the content of each."
        )
