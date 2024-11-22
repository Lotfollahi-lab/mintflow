
'''
Utilities for multi-slice setting where there are >1 slices and separate entities (neighbourhood graphs, dataloaders, etc) has to be created for each slice,
while some info is still shared among all slices, like the set of cell types.
'''

from typing import List
import scanpy as sc
import torch


class Slice:
    def __init__(
        self,
        adata:sc.AnnData,
        dict_obskey:dict,
        kwargs_compute_graph:dict,
        flag_use_custompygsampler:bool,
        kwargs_pygdl_train:dict,
        kwargs_pygdl_test:dict
    ):
        '''

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
        ...

        '''
        self.adata = adata
        self.dict_obskey = dict_obskey
        self.kwargs_compute_graph = kwargs_compute_graph
        self.flag_use_custompygsampler = flag_use_custompygsampler
        self.kwargs_pygdl_train = kwargs_pygdl_train
        self.kwargs_pygdl_test = kwargs_pygdl_test
        self._check_args()




    def _check_args(self):

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


class ListSlice:
    def __init__(
        self,
        list_slice:List[Slice]
    ):
        self.list_slice = list_slice
        self._check_args()



    def _check_args(self):
        assert isinstance(self.list_slice, list)
        for u in self.list_slice:
            assert isinstance(u, Slice)




