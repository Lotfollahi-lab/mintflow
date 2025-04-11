'''
Utitlity for masking parts of a tissue to be imputed by MintFlow.
'''

import os, sys
import PIL
import PIL.Image
import anndata._core.anndata
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import torch
import scanpy as sc
import squidpy as sq


def func_img_to_masks(fname_img:str, num_regions_excludingthebackground:int, flag_verbose:bool):
    '''
    Given the png of manual masks, extracts/returns different layers of course after discarding the background.
    :param fname_img:
    :param num_regions_excludingthebackground
    :return:
    '''
    assert(os.path.isfile(fname_img))
    assert(fname_img.endswith(".png"))

    # read the image
    np_img = np.array(PIL.Image.open(fname_img))[:, :, 0:3] / 255.0
    np_img_reshaped = np.reshape(np_img + 0.0, [np_img.shape[0] * np_img.shape[1], 3])

    # apply kMeans
    kmeans = KMeans(n_clusters=num_regions_excludingthebackground + 1, random_state=0).fit(
        np_img_reshaped
    )
    labels = kmeans.predict(
        np_img_reshaped
    )
    labels_reshaped = np.reshape(labels, [np_img.shape[0], np_img.shape[1]])

    # identify the label corresponding to the white background ===
    c_background, mindist_to_white = None, np.inf
    for c in range(num_regions_excludingthebackground + 1):
        meancolor_c = np.mean(np_img_reshaped[labels == c, :], 0)
        currdist_to_white = np.sum(
            (meancolor_c - np.array([1.0, 1.0, 1.0])) * (meancolor_c - np.array([1.0, 1.0, 1.0]))
        )
        if currdist_to_white <= mindist_to_white:
            c_background = c
            mindist_to_white = currdist_to_white
            if flag_verbose:
                print("c background was set to {}, with mean color:{}".format(
                    c,
                    meancolor_c
                ))

    # show different regions ===
    if flag_verbose:
        plt.figure()
        plt.subplot(1, num_regions_excludingthebackground + 1, 1)
        plt.imshow(np_img)
        plt.axis('off')
        plt.title("Background label={}".format(c_background))
        cnt_plot = 2
        for c in range(num_regions_excludingthebackground + 1):
            if c != c_background:
                plt.subplot(1, num_regions_excludingthebackground + 1, cnt_plot);
                cnt_plot += 1
                plt.imshow(labels_reshaped == c)
                plt.title("Label {}".format(c))
                plt.axis('off')
        plt.show()

    # compute/ret the masks ===
    dict_localc_to_mask = {}
    for c in range(num_regions_excludingthebackground + 1):
        if c != c_background:
            dict_localc_to_mask[c] = (labels_reshaped == c)+0.0

    # show the final retval
    if flag_verbose:
        plt.figure()
        plt.subplot(1, num_regions_excludingthebackground + 1, 1)
        plt.imshow(np_img)
        plt.axis('off')
        plt.title("Background label={}".format(c_background))
        cnt_plot = 2
        for localc in dict_localc_to_mask.keys():
            plt.subplot(1, num_regions_excludingthebackground + 1, cnt_plot);
            cnt_plot += 1
            plt.imshow(dict_localc_to_mask[localc], cmap='magma')
            plt.title("localkey {}".format(localc))
            plt.axis('off')
        plt.show()

    if flag_verbose:
        for localc in dict_localc_to_mask.keys():
            print("localkey={} --> mask in [{}, {}]".format(
                localc,
                np.min(dict_localc_to_mask[localc]),
                np.max(dict_localc_to_mask[localc])
            ))


    return dict_localc_to_mask

@torch.no_grad()
def func_dictmasks_to_cellmasklabels(
    dict_localc_to_mask:dict,
    ten_xy_absolute:torch.Tensor,
    flag_verbose:bool,
    adata:anndata._core.anndata.AnnData,
    kwargs_scatter_wholetissue:dict
):
    '''
    :param dict_localc_to_mask: the mask dict as returned by `func_img_to_masks`
    :param ten_xy_absolute
    :return:
    '''
    assert(
        isinstance(adata, anndata._core.anndata.AnnData)
    )

    # infer hw of masks
    h, w = None, None
    for localc in dict_localc_to_mask.keys():
        if h is None:
            h = dict_localc_to_mask[localc].shape[0]
            w = dict_localc_to_mask[localc].shape[1]
        else:
            assert (h == dict_localc_to_mask[localc].shape[0])
            assert (w == dict_localc_to_mask[localc].shape[1])

    # make ten_x and ten_y in the pixel coordinate (as opposed to the original xy coordinates).
    ten_x = ten_xy_absolute[:, 0] + 0.0  # [num_cells]
    ten_y = ten_xy_absolute[:, 1] + 0.0  # [num_cells]
    ten_x = ten_x - ten_x.min()
    ten_x = ten_x/ten_x.max()
    ten_y = ten_y - ten_y.min()
    ten_y = ten_y / ten_y.max()
    ten_x = ten_x * w  # [num_cells] between 0 and w
    ten_y = ten_y * h  # [num_cells] between 0 and h
    ten_x = torch.clamp(
        torch.clamp(ten_x, min=0, max=w-1).int(),
        min=0,
        max=w-1
    ).int()  # [num_cells] between 0 and w
    ten_y = torch.clamp(
        torch.clamp(ten_y, min=0, max=h-1).int(),
        min=0,
        max=h-1
    ).int()  # [num_cells] between 0 and h

    # create `dict_localc_to_cellbinarylabel` where for each localc gives a binary label.
    list_localc = list(dict_localc_to_mask.keys())
    list_localc.sort()
    dict_newc_to_cellbinarylabel = {
        1+list_localc.index(localc):dict_localc_to_mask[localc][
            ten_y.detach().cpu().numpy().tolist(),
            ten_x.detach().cpu().numpy().tolist()
        ] > 0.0
        for localc in dict_localc_to_mask.keys()
    }  # if newc>1 --> the mask if for one of the manual polygons and newc=0 corresponds to the training mask for model.
    assert (0 not in dict_newc_to_cellbinarylabel.keys())
    dict_newc_to_cellbinarylabel[0] = True
    for newc in dict_newc_to_cellbinarylabel.keys():
        if newc > 0:
            dict_newc_to_cellbinarylabel[0] = dict_newc_to_cellbinarylabel[0] * ~dict_newc_to_cellbinarylabel[newc]

    # add the masking labels to adata
    for newc in dict_newc_to_cellbinarylabel.keys():
        adata.obs['mintflow_maskinglabel_{}'.format(newc)] = \
            [str(u) for u in (dict_newc_to_cellbinarylabel[newc]+0.0).astype(int)]

    if flag_verbose:
        for c in range(1+max(dict_newc_to_cellbinarylabel.keys())):
            sq.pl.spatial_scatter(
                adata,
                spatial_key='spatial',
                img=False,
                connectivity_key="spatial_connectivities",
                library_id='connectivities_key',  # 'connectivities_key',
                color='mintflow_maskinglabel_{}'.format(c),
                crop_coord=None,
                **kwargs_scatter_wholetissue
            )

    return dict_newc_to_cellbinarylabel

