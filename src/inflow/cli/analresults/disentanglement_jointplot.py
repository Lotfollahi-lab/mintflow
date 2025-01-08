
"""
Shows the joint plot (disentanglment across different count values).
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def vis(
    adata_unnorm,
    pred_Xspl_rownormcorrected,
    list_LR,
    fname_dump
):
    """
    :param adata_unnorm:
    :param pred_Xspl_rownormcorrected:
    :param list_LR: the list of genes found both in the LR databse and the gene pannel.
    :param fname_dump:
    :return:
    """

    for g in list_LR:
        assert g in adata_unnorm.var.index.tolist()

    list_geneindex_inLR = [
        adata_unnorm.var.index.tolist().index(g) for g in list_LR
    ]
    list_geneindex_inLR.sort()
    cnt_thresh_x_obs = 0  # dummy filter atm


    mask_inLR = adata_unnorm.X.toarray()[:, list_geneindex_inLR] > cnt_thresh_x_obs
    mask_notinLR = adata_unnorm.X.toarray()[:,
                   list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))] > cnt_thresh_x_obs
    mask_all = adata_unnorm.X.toarray() > cnt_thresh_x_obs

    plt.figure()
    red_x = adata_unnorm.X.toarray()[:, list_geneindex_inLR][mask_inLR].flatten()
    red_y = pred_Xspl_rownormcorrected[:, list_geneindex_inLR][mask_inLR].flatten()
    sns.jointplot(
        data=pd.DataFrame(
            np.stack([red_x, red_y], -1),
            columns=['readout counts', 'predicted in predXspl']
        ),
        x="readout counts",
        y='predicted in predXspl',
        color='r',
        kind="scatter"
    )
    plt.xlim(
        adata_unnorm.X.toarray()[mask_all].flatten().min(),
        adata_unnorm.X.toarray()[mask_all].flatten().max()
    )
    plt.ylim(
        pred_Xspl_rownormcorrected[mask_all].flatten().min(),
        pred_Xspl_rownormcorrected[mask_all].flatten().max()
    )

    plt.xlabel("observed count X_obs")
    plt.ylabel("predicted X_spl")
    plt.title("Among Columns of adata.X \n (i.e. genes) found in LR database")

    # plt.show()
    plt.savefig(
        fname_dump
    )
    plt.close()


