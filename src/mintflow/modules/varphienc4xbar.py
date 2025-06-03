

'''
Encoder module for x-->xbar. The \varphi(.) functions with the notaion of paper.
Batch token is concatenated if `flag_enable_batchEmb` is set to True, otherwise all-zero vectors are concatenated.
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

        '''
        if self.flag_enable_batchEmb:
            self.param_batchshift = nn.Parameter(
                torch.randn(
                    [self.num_batches-1, self.dim_xbar],
                    requires_grad=True
                ),
                requires_grad=True
            )  # [num_batches-1 x dim_xbar] one minus num_batches because the 1st batch is considered as the reference --> no shift for the 1st batch.
            # TODO: upperbound the shift by 1. tanh(.) layer followed by 2. some bounded coefficients.
        '''

        # if self.num_batches == 1:
        #     raise NotImplementedError(
        #         "Not implemented for 1 batch. TODO: makes the batch token non-trainable and zero in that case."
        #     )


    def forward(self, x, batch):
        """
        :param x: a tensor of shape [N x num_gene]
        :param batch: pyg.NeighbourLoader batch, `batch.y` is to be used to get batch embeddings.
        :return: xbar
        """

        if self.flag_enable_batchEmb:
            assert (
                batch.y.size()[1] == (batch.MINTFLOWMETAINF['dim_u_int'] + batch.MINTFLOWMETAINF['dim_u_spl'] + batch.MINTFLOWMETAINF['dim_CT'] + batch.MINTFLOWMETAINF['dim_NCC'] + batch.MINTFLOWMETAINF['dim_BatchEmb'])
            )
            rng_batchEmb = [
                batch.MINTFLOWMETAINF['dim_u_int'] + batch.MINTFLOWMETAINF['dim_u_spl'] + batch.MINTFLOWMETAINF['dim_CT'] + batch.MINTFLOWMETAINF['dim_NCC'],
                batch.MINTFLOWMETAINF['dim_u_int'] + batch.MINTFLOWMETAINF['dim_u_spl'] + batch.MINTFLOWMETAINF['dim_CT'] + batch.MINTFLOWMETAINF['dim_NCC'] + batch.MINTFLOWMETAINF['dim_BatchEmb']
            ]
            ten_batchEmb = batch.y[
                :,
                rng_batchEmb[0]:rng_batchEmb[1]
            ].float().to(x.device).detach()  # [N x num_batches], the one-hot encoded batch token.
            assert ten_batchEmb.size()[1] == self.num_batches
        else:
            ten_batchEmb = torch.zeros([x.size()[0], self.num_batches]).float().to(x.device).detach()  # [N x num_batches], all-zero vectors.

        if len(ten_batchEmb.size()) == 1:
            ten_batchEmb = ten_batchEmb.unsqueeze(-1)  # in the case of 1 batch --> avoid the dim issue.

        output = self.module_encX(
            torch.cat([ten_batchEmb, x], 1)
        )  # [N x dim_xbar]

        return output



    def _check_args(self):
        assert isinstance(self.module_encX, nn.Module)
        assert isinstance(self.num_batches, int)
        assert self.num_batches > 0
        assert isinstance(self.flag_enable_batchEmb, bool)
        assert isinstance(self.dim_xbar, int)




