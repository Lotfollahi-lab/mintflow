


"""
Shows the violint plots, disentanglment across a range of different count values.
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.autonotebook import tqdm

def vis(
    adata_unnorm,
    pred_Xspl_rownormcorrected,
    min_cnt_vertical_slice,
    max_cnt_vertical_slice,
    list_LR,
    path_dump,
    str_sampleID,
    str_batchID,
    idx_slplus1
):
    """
    :param adata_unnorm:
    :param pred_Xspl_rownormcorrected:
    :param list_LR: the list of genes found both in the LR databse and the gene pannel.
    :param fname_dump_red:
    :param fname_dump_blue
    :return:
    """

    assert adata_unnorm.shape[0] == pred_Xspl_rownormcorrected.shape[0]
    assert adata_unnorm.shape[1] == pred_Xspl_rownormcorrected.shape[1]

    for g in list_LR:
        assert g in adata_unnorm.var.index.tolist()

    list_geneindex_inLR = [
        adata_unnorm.var.index.tolist().index(g) for g in list_LR
    ]
    list_geneindex_inLR.sort()

    for cnt_vertical_slice in tqdm(range(min_cnt_vertical_slice, max_cnt_vertical_slice), desc="Creating violin plots for slide {}".format(idx_slplus1)):

        mask_inLR = adata_unnorm.X.toarray()[:, list_geneindex_inLR] == cnt_vertical_slice
        mask_notinLR = adata_unnorm.X.toarray()[:, list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))] == cnt_vertical_slice
        mask_all = adata_unnorm.X.toarray() == cnt_vertical_slice

        slice_pred_inLR = pred_Xspl_rownormcorrected[:, list_geneindex_inLR][mask_inLR].flatten()
        slice_pred_notinLR = pred_Xspl_rownormcorrected[:, list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))][mask_notinLR].flatten()

        plt.figure()
        sns.violinplot(
            data={
                'not in LR-DB': slice_pred_notinLR / (cnt_vertical_slice + 0.0),
                'in LR-DB': slice_pred_inLR / (cnt_vertical_slice + 0.0),
            },
            cut=0
        )
        plt.title(
            "sample: {} \n in biological batch {} \n among readout counts={}".format(
                str_sampleID,
                str_batchID,
                cnt_vertical_slice
            )
        )
        plt.savefig(
            os.path.join(
                path_dump,
                '{}.png'.format(cnt_vertical_slice)
            ),
            bbox_inches='tight',
            pad_inches=0
        )
        plt.close()

        # plt.show()



