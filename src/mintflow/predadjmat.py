# When adding the batch-token, the adjmatpredictor-s work with `batch.edge_index` (and occasionaly `batch.x`).
# So they aren't affected by `batch.BatchEmb`.

'''
Utils for the Adjacancy matrix predictor losses.
'''
import numpy as np
from typing import List
import torch
import torch.nn as nn
import torch_geometric as pyg


class GradReverse(torch.autograd.Function):
    '''
    Code grabbed from: https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3
    '''
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

def func_all_pairwise_concatenations(x1: torch.Tensor, x2: torch.Tensor, flag_doublecheck: bool = False):
    '''
    :param x1: a tensor of shape [N, D1]
    :param x2: a tensor of shape [N, D2]
    :param flag_doublecheck:
    :return: output, a tensor of shape [N*N, D1+D2], the concatenation of each two pairs in x1 and x2.
    '''
    assert (
        x1.size()[0] == x2.size()[0]
    )
    N = x1.size()[0]
    D1 = x1.size()[1]
    D2 = x2.size()[1]

    # make part 1
    output_part_1 = torch.repeat_interleave(x1, repeats=N, dim=0)  # [N*N, D1]

    # make part 2
    output_part_2 = torch.tile(x2, [N, 1])  # [N*N, D2]


    output = torch.cat([output_part_1, output_part_2], 1) + 0.0  # [N*N, D1+D2]


    if flag_doublecheck:
        with torch.no_grad():
            for n1 in range(N):
                for n2 in range(N):
                    assert (
                        torch.any(
                            output[n1 * N + n2, :] == torch.cat(
                                (x1[n1, :].flatten(), x2[n2, :].flatten()),
                                0
                            )
                        ).item()
                    )
    return output


class AdjMatPredLoss(nn.Module):
    def __init__(
        self,
        module_predictor:nn.Module,
        varname_1:str,
        varname_2:str,
        str_predicable_or_unpredictable:str,
        coef_loss:float,
        flag_defineloss_onlyon_pygneighinternal:bool
    ):
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
        self.coef_loss = coef_loss
        self.flag_defineloss_onlyon_pygneighinternal = flag_defineloss_onlyon_pygneighinternal
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        self._check_args()

    def _check_args(self):
        assert (
            self.flag_defineloss_onlyon_pygneighinternal in [False, True]
        )
        assert (
            isinstance(self.coef_loss, float)
        )
        assert (self.coef_loss > 0.0)
        assert(
            isinstance(self.str_predicable_or_unpredictable, str)
        )
        assert(
            self.str_predicable_or_unpredictable in ['predictable', 'unpredictable']
        )
        module_lastchild = list(self.module_predictor.children())[-1]
        assert (
            isinstance(module_lastchild, nn.Linear)
        )
        assert (
            isinstance( list(self.module_predictor.children())[0], nn.Linear)
        )  # to make sure that the called doesn't include GRL, because GRL is added locally here.
        assert (
            module_lastchild.out_features == 2
        )

    def forward(self, dict_q_sample, pyg_batch):
        '''

        :param dict_q_sample: as returned by `MintFlowVarDist.rsample`
        :param pyg_batch: a mini-batch returned by pyg's negihbourloader.
        :return:
        '''
        # compute the dense adjecancy matrix to build the adjpred loss.
        with torch.no_grad():
            dense_adj = pyg.utils.to_dense_adj(
                pyg_batch.edge_index.to(dict_q_sample['xbar_int'].device),
                batch=torch.Tensor([0 for u in range(pyg_batch.x.shape[0])]).long().to(dict_q_sample['xbar_int'].device)
            )[0, :, :]  # [bsize, bsize]
            dense_adj = dense_adj.to(dict_q_sample['xbar_int'].device)
            '''
            To double-check
            for colidx in range(pyg_batch.edge_index.size()[1]):
                print(colidx)
                assert (
                    dense_adj[pyg_batch.edge_index[0, colidx], pyg_batch.edge_index[1, colidx]] == 1
                )
            '''
            dense_adj = ((dense_adj + dense_adj.T) > 0.0) + 0.0  # [bsize, bsize]
            dense_adj = dense_adj * (1.0 - torch.eye(pyg_batch.x.size()[0], device=dense_adj.device))  # [bsize, bsize]

        var_input_1 = dict_q_sample[self.varname_1]  # [bsize, dimvar1]
        var_input_2 = dict_q_sample[self.varname_2]  # [bsize, dimvar2]
        ten_var1var2 = func_all_pairwise_concatenations(
            x1=var_input_1,
            x2=var_input_2,
            flag_doublecheck=False
        )  # [bsize*bsize, dimvar1+dimvar2] # TODO: revert flag_doublecheck to False for speed-up



        if self.str_predicable_or_unpredictable == 'unpredictable':
            netout = self.module_predictor(
                grad_reverse(ten_var1var2)
            )  # [bsize*bsize, 2]
        elif self.str_predicable_or_unpredictable == 'predictable':
            netout = self.module_predictor(
                ten_var1var2
            )  # [bsize*bsize, 2]
        else:
            raise Exception(
                "Unknown str_predicable_or_unpredictable = {}".format(self.str_predicable_or_unpredictable)
            )

        loss = self.celoss(netout, dense_adj.flatten().long())  # [bsize*bsize]
        loss = loss.reshape(pyg_batch.x.size()[0], pyg_batch.x.size()[0])  # [bsize, bsize]

        if self.flag_defineloss_onlyon_pygneighinternal:
            loss = loss[0:pyg_batch.batch_size, :]
            loss = loss[:, 0:pyg_batch.batch_size]  # [intnode, intnode]
            num_defloss = pyg_batch.batch_size + 0.0
        else:
            num_defloss = pyg_batch.x.size()[0]




        loss = torch.triu(loss, diagonal=1)  # [num_defloss, num_defloss]
        loss = torch.sum(loss) / (0.5 * num_defloss * (num_defloss - 1.0))

        return loss * self.coef_loss



class ListAdjMatPredLoss(nn.Module):
    def __init__(self, list_adjpredictors:List[AdjMatPredLoss]):
        super(ListAdjMatPredLoss, self).__init__()
        assert (
            isinstance(list_adjpredictors, list)
        )
        for u in list_adjpredictors:
            assert (isinstance(u, AdjMatPredLoss))

        self.list_adjpredictors = nn.ModuleList(list_adjpredictors)

    def forward(self, dict_q_sample, pyg_batch):
        '''
        :param dict_q_sample: as returned by `MintFlowVarDist.rsample`
        :param pyg_batch: a mini-batch returned by pyg's negihbourloader.
        :return:
        '''

        dict_varname_to_loss = {
            "{}&{}".format(adjpredictor.varname_1, adjpredictor.varname_2):adjpredictor.forward(dict_q_sample, pyg_batch) \
            for adjpredictor in self.list_adjpredictors
        }
        return dict_varname_to_loss






