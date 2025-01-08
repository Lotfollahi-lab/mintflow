
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
    fname_dump,

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    # red =======================
    red_x = adata_unnorm.X.toarray()[:, list_geneindex_inLR][mask_inLR].flatten()
    red_y = pred_Xspl_rownormcorrected[:, list_geneindex_inLR][mask_inLR].flatten()
    sns.jointplot(
        ax=axes[0],
        data=pd.DataFrame(
            np.stack([red_x, red_y], -1),
            columns=['readout counts', 'predicted in predXspl']
        ),
        x="readout counts",
        y='predicted in predXspl',
        color='r',
        kind="scatter"
    )
    axes[0].set_xlim(
        adata_unnorm.X.toarray()[mask_all].flatten().min(),
        adata_unnorm.X.toarray()[mask_all].flatten().max()
    )
    axes[0].set_ylim(
        pred_Xspl_rownormcorrected[mask_all].flatten().min(),
        pred_Xspl_rownormcorrected[mask_all].flatten().max()
    )

    axes[0].set_xlabel("observed count X_obs")
    axes[0].set_ylabel("predicted X_spl")
    axes[0].set_title("Among Columns of adata.X \n (i.e. genes) found in the LR database.")

    # blue =======================
    blue_x = adata_unnorm.X.toarray()[:, list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))][mask_notinLR].flatten()
    blue_y = pred_Xspl_rownormcorrected[:, list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))][mask_notinLR].flatten()
    sns.jointplot(
        ax=axes[1],
        data=pd.DataFrame(
            np.stack([blue_x, blue_y], -1),
            columns=['readout counts', 'predicted in predXspl']
        ),
        x="readout counts",
        y='predicted in predXspl',
        color='b',
        kind="scatter"
    )
    axes[1].set_xlim(
        adata_unnorm.X.toarray()[mask_all].flatten().min(),
        adata_unnorm.X.toarray()[mask_all].flatten().max()
    )
    axes[1].set_ylim(
        pred_Xspl_rownormcorrected[mask_all].flatten().min(),
        pred_Xspl_rownormcorrected[mask_all].flatten().max()
    )
    axes[1].set_xlabel("observed count X_obs")
    axes[1].set_ylabel("predicted X_spl")
    axes[1].set_title("Among Columns of adata.X \n (i.e. genes) not found in the LR database.")

    # plt.show()
    fig.savefig(
        fname_dump
    )
    plt.close(fig)


