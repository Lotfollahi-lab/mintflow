'''
Utils for the Adjacancy matrix predictor losses.
'''
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch_geometric as pyg


def func_all_pairwise_concatenations(x1:torch.Tensor, x2:torch.Tensor, flag_doublecheck:bool=False):
    '''

    :param x1: a tensor of shape [N, D1]
    :param x2: a tensor of shape [N, D2]
    :return: output, a tensor of shape [N*N, D1+D2], the concatenation of each two pairs in x1 and x2.
    '''
    pass
    assert(
        x1.size()[0] == x2.size()[0]
    )
    N = x1.size()[0]
    D1 = x1.size()[1]
    D2 = x2.size()[1]

    output = x1.view(N, 1, D1, 1) + x2.view(1, N, 1, D2) + 0.0  # [N, N, D1, D2]
    output = output.reshape(N*N, D1, D2) + 0.0  # [N*N, D1, D2]
    output = output.reshape(N*N, D1+D2) + 0.0  # [N*N, D1+D2]

    if flag_doublecheck:
        with torch.no_grad():
            for n1 in range(N):
                for n2 in range(N):
                    assert(
                        torch.all(
                            output[n1*N + n2, :] == torch.cat(
                                (x1[n1,:].flatten(), x2[n2,:].flatten()),
                                0
                            )
                        ).item()
                    )
    return output


class AdjMatPredLoss(nn.Module):
    def __init__(self, module_predictor:nn.Module, varname_1:str, varname_2:str, str_predicable_or_unpredictable:str):
        '''

        :param module_predictor: the predictor module. Since a cross-entropy loss is to be placed, this modules
            has to end with `nn.Linear(..., dim_output=2, ...)`
        :param varname_1, varname_2: the prediction loss will be created for very pairs of cells, where varname_1 of cell1 and varname_2 of cell2 are concatenated
            and the edge between cell1 and cell2 are to be predicted.
        :param str_predicable_or_unpredictable:
        '''
        super(AdjMatPredLoss, self).__init__()
        self.module_predictor = module_predictor
        self.varname_1 = varname_1
        self.varname_2 = varname_2
        self.str_predicable_or_unpredictable = str_predicable_or_unpredictable
        self.celoss = nn.CrossEntropyLoss()
        self._check_args()

    def _check_args(self):
        assert(
            isinstance(self.str_predicable_or_unpredictable, str)
        )
        assert(
            self.str_predicable_or_unpredictable in ['predictable', 'unpredictable']
        )
        module_lastchild = list(self.module_predictor.children())[-1]
        assert(
            isinstance(module_lastchild, nn.Linear)
        )
        assert(
            module_lastchild.out_features == 2
        )

    def forward(self, dict_q_sample, pyg_batch):
        '''

        :param dict_q_sample: as returned by `InFlowVarDist.rsample`
        :param pyg_batch: a mini-batch returned by pyg's negihbourloader.
        :return:
        '''
        # compute the dense adjecancy matrix to build the adjpred loss.
        with torch.no_grad():
            dense_adj = pyg.utils.to_dense_adj(
                pyg_batch.edge_index,
                batch=torch.Tensor([0 for u in range(pyg_batch.x.shape[0])]).long()
            )[0, :, :]  # [bsize, bsize]
            '''
            To double-check
            for colidx in range(pyg_batch.edge_index.size()[1]):
                print(colidx)
                assert (
                    dense_adj[pyg_batch.edge_index[0, colidx], pyg_batch.edge_index[1, colidx]] == 1
                )
            '''
            dense_adj = ((dense_adj + dense_adj.T) > 0.0) + 0.0  # [bsize, bsize]
            dense_adj = dense_adj * (1.0 - torch.eye(pyg_batch.x.size()[0]))  # [bsize, bsize]

        var_input_1 = dict_q_sample[self.varname_1]  # [bsize, dimvar]
        var_input_2 = dict_q_sample[self.varname_2]  # [bsize, dimvar]


        netout = self.module_predictor(x_input)  # [bsize, 2]
        loss = self.celoss(netout, dense_adj)









