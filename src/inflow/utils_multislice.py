
'''
Utilities for multi-slice setting where there are >1 slices and separate entities (neighbourhood graphs, dataloaders, etc) has to be created for each slice,
while some info is still shared among all slices, like the set of cell types.
'''

from typing import List
import numpy as np
import gc
import squidpy as sq
import scanpy as sc
import torch_geometric as pyg
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import torch
from torch_geometric.loader import NeighborLoader
from . import utils_pyg

from . import modules
from .modules import gnn


class Slice:
    def __init__(
        self,
        adata:sc.AnnData,
        dict_obskey:dict,
        kwargs_compute_graph:dict,
        flag_use_custompygsampler:bool,
        kwargs_pygdl_train:dict,
        kwargs_pygdl_test:dict,
        neighgraph_num_hops_computeNCC:int,
        batchsize_compute_NCC:int,
        device,
        kwargs_sq_pl_spatial_scatter:dict = None
    ):
        """

        :param adata: the anndata corresponding to "only" the slice of concernt.
        :param dict_obskey: a dictionary containing the column names of interest in the input anndata
        Keys
            - x: the x position
            - y: the y position
            - cell_type: to be used to obtain, CT and NCC tensors.
            - sliceid_to_checkUnique: the ID of each slice, to double check they are unique in the anndata list.
        :param kwargs_compute_graph: kwargs to pass to `sq.gr.spatial_neighbors`
        :param flag_use_custompygsampler: whether to use the custom (i.e. window-based) sampler for pyg.NeighbourLoader.
        :param kwargs_pygdl_train: there are two cases
        - flag_use_custompygsampler is set to True. In this case this dict contains
        --- num_neighbors: List[int]
        --- width_window: int (for the custom sampler)
        --- min_numpoints_ingrid: (for the custom sampler)
        --- flag_disable_randoffset (for the custom sampler)
        - flag_use_custompygsampler is set to False. In this case this dict contains
        --- num_neighbors: List[int]
        --- batch_size: int
        :param kwargs_pygdl_test: there are two cases (same as above)
        :param neighgraph_num_hops_computeNCC: number of hopes in the neighbourhood graph, to compute NCC vectors from CT vectors.
        :param batchsize_compute_NCC: the batch-size to compute NCC vectors from CT vectors.
        :param device: the device to be used to compute NCC vectors (i.e. neighbourhoold cell type composition) from cell types.
        :param kwargs_sq_pl_spatial_scatter: the kwargs to show the scatter using squidpy. Optional, default=None.
        ...

        """
        self.adata = adata
        self.dict_obskey = dict_obskey
        self.kwargs_compute_graph = kwargs_compute_graph
        self.flag_use_custompygsampler = flag_use_custompygsampler
        self.kwargs_pygdl_train = kwargs_pygdl_train
        self.kwargs_pygdl_test = kwargs_pygdl_test
        self.neighgraph_num_hops_computeNCC = neighgraph_num_hops_computeNCC
        self.batchsize_compute_NCC = batchsize_compute_NCC
        self.device = device
        self.kwargs_sq_pl_spatial_scatter = kwargs_sq_pl_spatial_scatter

        self._check_args()

    @torch.no_grad()
    def _add_CT_NCC(self):
        """
        Adds
        - `self.ten_CT`
        - `self.ten_NCC`
        :return:
        """

        list_celltype_int = []
        for idx_row in range(self.adata.shape[0]):
            str_ct = self.adata.obs['inflow_CT'].iloc[idx_row]
            assert str_ct[0:len('inflowCT_')] == 'inflowCT_'
            list_celltype_int.append(
                int(str_ct.split("_")[1])
            )

        ten_CT = torch.eye(self._global_num_CT)[list_celltype_int, :] + 0.0
        module_compNCC = modules.gnn.KhopAvgPoolWithoutselfloop(
            num_hops=self.neighgraph_num_hops_computeNCC,
            dim_input=None,
            dim_output=None
        )
        module_compNCC = module_compNCC.to(self.device)

        ten_NCC = module_compNCC.evaluate_layered(
            x=(ten_CT + 0.0).float().to(self.device),
            edge_index=(self.edge_index + 0).to(self.device),
            kwargs_dl={
                'batch_size': self.batchsize_compute_NCC,
                'num_workers': 0,
                'num_neighbors': [-1]
            }
        )

        del module_compNCC
        gc.collect()

        self.ten_CT  = ten_CT.to("cpu")
        self.ten_NCC = ten_NCC.to("cpu")

    @torch.no_grad()
    def _add_pygds_pygdl(self):
        """
        Creates
        - `self.pyg_ds`
        - `self.pyg_dl_train`
        - `self.pyg_dl_test`
        - `self.ten_xy_absolute`

        :return:
        """

        ten_u_z = self.ten_CT + 0.0
        ten_u_s = self.ten_CT + 0.0

        key_x, key_y = self.dict_obskey['x'], self.dict_obskey['y']
        self.ten_xy_absolute = torch.tensor(
            np.stack(
                [np.array(self.adata.obs[key_x].tolist()), np.array(self.adata.obs[key_y].tolist())],
                1
            )
        ).float().to(self.device)


        self.pyg_ds = pyg.data.Data(
            x=torch.sparse_coo_tensor(
                indices=self.adata.X.tocoo().nonzero(),
                values=self.adata.X.tocoo().data,
                size=self.adata.X.tocoo().shape
            ).float(),
            edge_index=self.edge_index,
            y=torch.cat(
                [ten_u_z, ten_u_s, self.ten_CT, self.ten_NCC],
                1
            )
        )

        if self.adata.X.shape[0] * self.adata.X.shape[1] < 1e9:
            assert (
                torch.all(
                    torch.isclose(self.pyg_ds.x.to_dense(), torch.tensor(self.adata.X.toarray()).float())
                )
            )
            print("assertion was passed.")

        if self.flag_use_custompygsampler:
            print("Using the custom sampler for pygloader.")
            self.pyg_dl_train = NeighborLoader(
                self.pyg_ds,
                num_neighbors=self.kwargs_pygdl_train['num_neighbors'],
                batch_sampler=utils_pyg.PygSTDataGridBatchSampler(
                    ten_xy=self.ten_xy_absolute,
                    width_window=self.kwargs_pygdl_train['width_window'],
                    min_numpoints_ingrid=self.kwargs_pygdl_train['min_numpoints_ingrid'],
                    flag_disable_randoffset=self.kwargs_pygdl_train['flag_disable_randoffset']
                ),
                num_workers=0
            )
            self.pyg_dl_test = NeighborLoader(
                self.pyg_ds,
                num_neighbors=self.kwargs_pygdl_test['num_neighbors'],
                batch_sampler=utils_pyg.PygSTDataGridBatchSampler(
                    ten_xy=self.ten_xy_absolute,
                    width_window=self.kwargs_pygdl_test['width_window'],
                    min_numpoints_ingrid=self.kwargs_pygdl_test['min_numpoints_ingrid'],
                    flag_disable_randoffset=self.kwargs_pygdl_test['flag_disable_randoffset']
                ),
                num_workers=0
            )

            assert not self.kwargs_pygdl_train['flag_disable_randoffset']
            assert self.kwargs_pygdl_test['flag_disable_randoffset']

        else:
            print("using the default sampler.")
            self.pyg_dl_train = NeighborLoader(
                self.pyg_ds,
                num_neighbors=self.kwargs_pygdl_train['num_neighbors'],
                batch_size=self.kwargs_pygdl_train['batch_size'],
                shuffle=True,
                num_workers=0
            )
            self.pyg_dl_test = NeighborLoader(
                self.pyg_ds,
                num_neighbors=self.kwargs_pygdl_test['num_neighbors'],
                batch_size=self.kwargs_pygdl_test['batch_size'],
                shuffle=False,
                num_workers=0
            )


    def _show_scatter(self):
        if self.kwargs_sq_pl_spatial_scatter is None:
            raise Exception(
                "To call this funciton, the arg `kwargs_sq_pl_spatial_scatter` should be passed in.\n"+\
                "For more info reffer to the documentation for TODO."
            )

        self.adata.uns['spatial'] = self.adata.uns['spatial_neighbors']
        crop_coord = None
        sq.pl.spatial_scatter(
            self.adata,
            spatial_key='spatial',
            img=False,
            connectivity_key="spatial_connectivities",
            library_id='connectivities_key',  # 'connectivities_key',
            color=[
                "inflow_CT",
            ],
            crop_coord=crop_coord,
            **self.kwargs_sq_pl_spatial_scatter
        )

    def _get_set_CT(self):
        """
        Returns the set of cell type labels.
        :return:
        """
        return set(
            self.adata.obs[
                self.dict_obskey['cell_type']
            ].tolist()
        )

    def _set_global_num_CT(self, global_num_CT:int):
        self._global_num_CT = global_num_CT + 0


    def _add_inflowCTcol(self, dict_rename):
        assert ("inflow_CT" not in self.adata.obs.columns)
        self.adata.obs['inflow_CT'] = self.adata.obs[self.dict_obskey['cell_type']].map(dict_rename)

    def _add_spatial_neighbours(self):
        """
        Creates `self.edge_index`, a tensor of shape [num_edges x 2].
        :return:
        """
        key_x, key_y = self.dict_obskey['x'], self.dict_obskey['y']
        self.adata.obsm['spatial'] = np.stack(
            [np.array(self.adata.obs[key_x].tolist()), np.array(self.adata.obs[key_y].tolist())],
            1
        )
        sq.gr.spatial_neighbors(
            adata=self.adata,
            **self.kwargs_compute_graph
        )

        with torch.no_grad():
            edge_index, _ = from_scipy_sparse_matrix(self.adata.obsp['spatial_connectivities'])  # [2, num_edges]
            edge_index = torch.Tensor(pyg.utils.remove_self_loops(pyg.utils.to_undirected(edge_index))[0])

            assert (
                torch.all(
                    torch.eq(
                        edge_index,
                        pyg.utils.to_undirected(edge_index)
                    )
                )
            )

            self.edge_index = edge_index



    def _check_args(self):

        if self.flag_use_custompygsampler:
            if self.kwargs_pygdl_train['width_window'] != self.kwargs_pygdl_test['width_window']:
                raise Exception(
                    "kwargs_pygdl_train['neighgraph_num_hops'] is not equal to self.kwargs_pygdl_test['width_window']."
                )

        if self.flag_use_custompygsampler:
            if self.kwargs_pygdl_train.keys() != {'num_neighbors', 'width_window', 'min_numpoints_ingrid', 'flag_disable_randoffset'}:
                raise Exception(
                    "The passed `kwargs_pygdl_train` has redundant or missing keys. Refer to documentation for requirements."
                )
            if self.kwargs_pygdl_test.keys() != {'num_neighbors', 'width_window', 'min_numpoints_ingrid', 'flag_disable_randoffset'}:
                raise Exception(
                    "The passed `kwargs_pygdl_test` has redundant or missing keys. Refer to documentation for requirements."
                )
        else:
            if self.kwargs_pygdl_train.keys() != {'num_neighbors', 'batch_size'}:
                raise Exception(
                    "The passed `kwargs_pygdl_train` has redundant or missing keys. Refer to documentation for requirements."
                )
            if self.kwargs_pygdl_test.keys() != {'num_neighbors', 'batch_size'}:
                raise Exception(
                    "The passed `kwargs_pygdl_test` has redundant or missing keys. Refer to documentation for requirements."
                )



        assert isinstance(self.adata, sc.AnnData)
        assert isinstance(self.dict_obskey, dict)
        assert isinstance(self.kwargs_compute_graph, dict)
        assert isinstance(self.flag_use_custompygsampler, bool)
        assert isinstance(self.neighgraph_num_hops_computeNCC, int)
        assert self.neighgraph_num_hops_computeNCC >= 1

        for k in ['x', 'y', 'cell_type', 'sliceid_to_checkUnique']:
            if k not in self.dict_obskey.keys():
                raise Exception(
                    "The input dictionary `dict_obskey` is expected to have {} key.".format(k)
                )

            if self.dict_obskey[k] not in self.adata.obs.columns.tolist():
                raise Exception(
                    "dict_obskey[{}] is set to {}, but {} is not among anndata.obs columns.".format(
                        k,
                        self.dict_obskey[k],
                        self.dict_obskey[k]
                    )
                )

        if len(set(self.adata.obs[self.dict_obskey['sliceid_to_checkUnique']])) > 1:
            raise Exception(
                "The anndata passed in has more than tissue IDs according to column {}. \n This may indicate not a single slice is passed in.".format(
                    self.dict_obskey['sliceid_to_checkUnique']
                )
            )


class ListSlice:
    def __init__(
        self,
        list_slice:List[Slice]
    ):
        self.list_slice = list_slice
        self._check_args()

        # make internals
        self._create_CTmapping_and_inflowCT()
        self._create_neighgraphs()
        self._create_CT_NCC_Vectors()
        self._add_pygds_pygdl()

    def _add_pygds_pygdl(self):
        for sl in self.list_slice:
            sl : Slice
            sl._add_pygds_pygdl()



    def _create_CT_NCC_Vectors(self):
        for sl in self.list_slice:
            sl : Slice
            sl._add_CT_NCC()


    def _create_neighgraphs(self):
        for sl in self.list_slice:
            sl._add_spatial_neighbours()

    def _create_CTmapping_and_inflowCT(self):
        """
        - Creates `self.map_CT_to_inflowCT`
        - Adds `inflow_CT` (i.e. inflow cell type) column to each anndata in the list.
        :return:
        """
        # create the mapping
        set_all_CT = []
        for sl in self.list_slice:
            set_all_CT = set_all_CT + list(sl._get_set_CT())

        set_all_CT = list(set(set_all_CT))
        set_all_CT.sort()

        self.map_CT_to_inflowCT = {
            ct:"inflowCT_{}".format(idx_ct)
            for idx_ct, ct in enumerate(set_all_CT)
        }

        # add the column "inflow_CT"
        for sl in self.list_slice:
            sl : Slice
            sl._add_inflowCTcol(self.map_CT_to_inflowCT)

        # set the global number of CTs
        for sl in self.list_slice:
            sl : Slice
            sl._set_global_num_CT(len(set_all_CT))



    def show_scatters(self):
        for sl in self.list_slice:
            sl._show_scatter()

    def _check_args(self):
        assert isinstance(self.list_slice, list)
        for u in self.list_slice:
            assert isinstance(u, Slice)




