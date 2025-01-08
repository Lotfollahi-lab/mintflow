
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
    fname_dump_red,
    fname_dump_blue,
    str_sampleID,
    str_batchID

):
    """
    :param adata_unnorm:
    :param pred_Xspl_rownormcorrected:
    :param list_LR: the list of genes found both in the LR databse and the gene pannel.
    :param fname_dump_red:
    :param fname_dump_blue
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



    # red =======================
    plt.figure()
    red_x = adata_unnorm.X.toarray()[:, list_geneindex_inLR][mask_inLR].flatten()
    red_y = pred_Xspl_rownormcorrected[:, list_geneindex_inLR][mask_inLR].flatten()
    g = sns.jointplot(
        data=pd.DataFrame(
            np.stack([red_x, red_y], -1),
            columns=['readout counts', 'predicted in predXspl']
        ),
        x="readout counts",
        y='predicted in predXspl',
        color='r',
        kind="scatter"
    )
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()
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
    plt.suptitle("Among Columns of adata.X \n (i.e. genes) found in the LR database.\n sample: {} \n in biological batch {}".format(str_sampleID, str_batchID), y=1)
    plt.savefig(
        fname_dump_red
    )
    plt.close()

    # blue =======================
    blue_x = adata_unnorm.X.toarray()[:, list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))][mask_notinLR].flatten()
    blue_y = pred_Xspl_rownormcorrected[:, list(set(range(adata_unnorm.shape[1])) - set(list_geneindex_inLR))][mask_notinLR].flatten()
    g = sns.jointplot(
        data=pd.DataFrame(
            np.stack([blue_x, blue_y], -1),
            columns=['readout counts', 'predicted in predXspl']
        ),
        x="readout counts",
        y='predicted in predXspl',
        color='b',
        kind="scatter"
    )
    g.ax_marg_x.remove()
    g.ax_marg_y.remove()

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
    plt.suptitle("Among Columns of adata.X \n (i.e. genes) not found in the LR database.\n sample: {} \n in biological batch {}".format(str_sampleID, str_batchID), y=1)

    # plt.show()
    plt.savefig(
        fname_dump_blue
    )
    plt.close()



