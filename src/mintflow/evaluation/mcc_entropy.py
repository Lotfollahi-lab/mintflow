

"""
Calculates the entropy of MCC vectors (microenvironment cell type composition).
Kozachenko-Leonenko estimate of entropy is calculated.
"""
import anndata
import squidpy as sq
import torch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch_geometric as pyg
import pandas as pd
import numpy as np
import cupy as cp
from cuml.neighbors import NearestNeighbors
from scipy.special import psi, gamma
from tqdm.autonotebook import tqdm


from .. import modules


# Grabbed/modified from google (AI mode)
def _gpu_kl_entropy(np_data, k_calcentropy):
    """
    Computes Shannon entropy for high-dim continuous variables on GPU.
    data: cupy.ndarray of shape (N, d)
    """
    assert isinstance(np_data, np.ndarray)
    data = cp.asarray(np_data)
    k = k_calcentropy
    
    N, d = data.shape
    
    # 1. Use cuML to find the distance to the k-th neighbor
    # We set n_neighbors=k+1 because the first neighbor is the point itself (dist=0)
    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(data)
    distances, _ = nn.kneighbors(data)
    
    # Extract the k-th neighbor distance (last column)
    # Ensure no zero distances to avoid log(0)
    eps = distances[:, -1]
    eps = cp.maximum(eps, 1e-15) 
    
    # 2. Constants (calculated on CPU or GPU)
    v_d = (cp.pi**(d/2)) / gamma(1 + d/2)
    term1 = psi(N) - psi(k)
    term2 = cp.log(v_d)
    
    # 3. Final summation using CuPy
    sum_log_eps = cp.sum(cp.log(eps))
    entropy = term1 + term2 + (d / N) * sum_log_eps
    
    return float(entropy)


def get_MCC_entropy(
    adata:anndata.AnnData,
    kwargs_neighbourhood_graph:dict,
    obskey_celltype:str,
    device,
    k_calcentropy:int,
    batch_size_computeMCC:int = 10,
):
    """
    Calculates the entropy-esimate of MCC vectors per cell type.
    
    :param adata: The input anndata object.
    :type adata: anndata.AnnData
    :param kwargs_neighbourhood_graph: kwargs to create the neighbourhood graph. This function recreates the neighbourhood graph internally.
    :type kwargs_neighbourhood_graph: dict
    :param obskey_celltype: The column in `.obs` containig cell type annotations.
    :type obskey_celltype: str
    :param device: device, e.g., cpu or gpu (recommeneded)
    :param k_calcentropy: The number of nearest neighbours used by the Kozachenko-Leonenko estimator. Default is 1, while one can use, e.g., 3 or 5.
    :type k_calcentropy: int
    :param batch_size_computeMCC: The batch size of pyg neighbourloader to calculate the MCC vectors.
    :type batch_size_computeMCC: int
    """
    
    # compute the neighrborhood graph
    adata.uns = {}
    adata.obsp = {}
    sq.gr.spatial_neighbors(
        adata=adata,
        **kwargs_neighbourhood_graph
    )

    # get `edge_index`
    with torch.no_grad():
        edge_index, _ = from_scipy_sparse_matrix(adata.obsp['spatial_connectivities'])  # [2, num_edges]
        edge_index = torch.Tensor(pyg.utils.remove_self_loops(pyg.utils.to_undirected(edge_index))[0])

    df_CT = pd.get_dummies(adata.obs[obskey_celltype])
    ten_CT = torch.tensor(np.array(df_CT) + 0.0, requires_grad=False)
    
    # compute MCC
    module_compMCC = modules.gnn.KhopAvgPoolWithoutselfloop(
        num_hops=1,
        dim_input=None,
        dim_output=None
    )
    module_compMCC = module_compMCC.to(device)
    ten_MCC = module_compMCC.evaluate_layered(
        x=ten_CT,
        edge_index=edge_index,
        kwargs_dl={
            'batch_size':batch_size_computeMCC,
            'num_workers':0,
            'num_neighbors':[-1]
        }
    )

    # compute the entropy values ct by ct
    dict_ct_to_MCCentropy = {}
    tmp_assert_rowsel = 0.0
    for ct in tqdm(set(adata.obs[obskey_celltype].tolist()), desc="Computing MCC entropy for different cell types"):
        list_rowsel = (adata.obs[obskey_celltype] == ct).tolist()
        
        tmp_assert_rowsel = tmp_assert_rowsel + np.array(list_rowsel) + 0.0

        dict_ct_to_MCCentropy[ct] = _gpu_kl_entropy(
            ten_MCC[list_rowsel, :].detach().cpu().numpy(),
            k_calcentropy=k_calcentropy
        )
    
    assert np.allclose(
        tmp_assert_rowsel,
        np.ones_like(tmp_assert_rowsel)
    )

    # create the df toret, to be used for, e.g., visualisation
    df = pd.DataFrame(
        {'cell_type':[k for k in dict_ct_to_MCCentropy.keys()], 'MCC_entropy':[v for _, v in dict_ct_to_MCCentropy.items()]}
    )
    df = df.sort_values(by='MCC_entropy', ascending=False)

    return dict_ct_to_MCCentropy, df