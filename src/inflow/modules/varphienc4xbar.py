

'''
Encoder module for x-->xbar. The \varphi(.) functions with the notaion of paper.
The output xbar-s are shifted by some batch-specific shifts, similar to scArches.
'''
from typing import Union, Callable, Tuple, Any, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle


class EncX2Xbar(nn.Module):
    def __init__(self, module_encX:nn.Module, num_batches:int, dim_xbar:int, flag_enable_batchEmb:bool):
        """

        :param module_encX: the module that takes in a gex vector and outputs xbar.
        :param num_batches: total number of batches, an integer.
        :param dim_xbar: dim of xbar (equal to dim_s and dim_z).
        :param flag_enable_batchEmb: whether conditioning on batch token is enabled.
        """
        super(EncX2Xbar, self).__init__()
        self.module_encX = module_encX
        self.num_batches = num_batches
        self.dim_xbar = dim_xbar
        self.flag_enable_batchEmb = flag_enable_batchEmb
        self._check_args()

        if self.flag_enable_batchEmb:
            self.param_batchshift = nn.Parameter(
                torch.randn(
                    [self.num_batches, self.dim_xbar],
                    requires_grad=True
                ),
                requires_grad=True
            )  # [num_batches x dim_xbar]


    def forward(self, x, batch):
        """
        :param x: a tensor of shape [N x num_gene]
        :param batch: pyg.NeighbourLoader batch, `batch.y` is to be used to get batch embeddings.
        :return: xbar
        """
        output = self.module_encX(x)  # [N x dim_xbar]

        if self.flag_enable_batchEmb:
            assert (
                batch.y.size()[1] == (batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb'])
            )
            rng_batchEmb = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb']
            ]
            ten_batchEmb = batch.y[
                :,
                rng_batchEmb[0]:rng_batchEmb[1]
            ]  # [N x num_batches], the one-hot encoded batch token.
            assert ten_batchEmb.size()[1] == self.num_batches

            output = output + torch.mm(ten_batchEmb.detach(), self.param_batchshift)

        return output



    def _check_args(self):
        assert isinstance(self.module_encX, nn.Module)
        assert isinstance(self.num_batches, int)
        assert isinstance(self.flag_enable_batchEmb, bool)
        assert isinstance(self.dim_xbar, int)




