'''
The version of the `ImputerAndDisentangler` where it only has the disengl part.
'''

import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from linformer_pytorch import Linformer, Padder

from .impanddisentgl import MaskLabel


class SubgraphEmbeddingImpAndDisengl(nn.Module):
    def __init__(self, num_genes, dim_embedding, dim_em_iscentralnode, dim_em_blankorobserved):
        '''
        :param num_genes: .
        :param dim_embedding: The dim of embedding for each cell, must be a multiple of 4.
        '''
        super(SubgraphEmbeddingImpAndDisengl, self).__init__()
        # grab args
        self.dim_embedding = dim_embedding
        self.dim_em_iscentralnode = dim_em_iscentralnode
        self.dim_em_blankorobserved = dim_em_blankorobserved
        assert(self.dim_embedding%4 == 0)
        # make internals
        self.encoder_x = nn.Linear(
            num_genes,
            dim_embedding,
            bias=False  # TODO: should it be True, or tunable?
        )  # the initial linear transformation on x (so pe can be added to it).
        self.embedding_iscentralnode = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.dim_em_iscentralnode
        )  # This emebedding tells whether the cell is among the central nodes returned by the Neighloader.
        self.embedding_blankorobserved = nn.Embedding(
            num_embeddings=2,
            embedding_dim=self.dim_em_blankorobserved
        )

    @torch.no_grad()
    def _position_encoding(self, batch, ten_xy_absolute:torch.Tensor):
        '''
        The positional embedding as done in cellplm paper.
        :return:
        '''
        ten_xy = ten_xy_absolute[batch.n_id.tolist(), :]+0.0  # [N, 2] where N is the size of subgraph.

        # min-max normaliztion to get xy in [0, 100]
        ten_xy = ten_xy - torch.min(ten_xy, 0).values.unsqueeze(0)  # [N, 2]
        ten_xy = ten_xy / torch.clamp(torch.max(ten_xy, 0).values.unsqueeze(0), min=0.0001, max=torch.inf) # [N, 2]
        ten_xy = 100.0*ten_xy  # [N,2] in [0,100]

        #  compute pe
        dby4 = self.dim_embedding//4
        denum = torch.exp(
            (torch.tensor(range(dby4)) / dby4) * np.log(10000.0)
        ).unsqueeze(0).to(ten_xy_absolute.device) # [1, dby4]
        x_sin = torch.sin(ten_xy[:, 0].unsqueeze(1) / denum)  # [N, bdy4]
        x_cos = torch.cos(ten_xy[:, 0].unsqueeze(1) / denum)  # [N, bdy4]
        y_sin = torch.sin(ten_xy[:, 1].unsqueeze(1) / denum)  # [N, bdy4]
        y_cos = torch.cos(ten_xy[:, 1].unsqueeze(1) / denum)  # [N, bdy4]
        pe = torch.cat(
            [x_sin, x_cos, y_sin, y_cos],
            1
        )  # [N, self.dim_embedding]
        return pe



    def forward(self, batch, prob_maskknowngenes:float, ten_xy_absolute:torch.Tensor):
        '''
        :param batch: the batch returned by pyg's neighborloader.
            - batch.x is a sparse matrix (e.g. adata.X).
            - batch.x must be raw counts (this function does log1p normalization) .
        :param prob_maskknowngenes: #TODO add the important documentation.
        :param ten_xy_absolute: the xy coordinates for the entire graph (not only the batch).
        :return:
        '''
        assert(ten_xy_absolute.size()[0] > batch.x.shape[0])
        x = torch.log(
            1.0 + batch.x.to_dense()
        ).to(ten_xy_absolute.device)
        ten_initmask = torch.tensor([False]*len(batch.y.tolist())).unsqueeze(0).to(ten_xy_absolute.device)
           # no imputation now --> no unobserved expvect. (batch.y == MaskLabel.UNKNOWN_TEST.value)
        with torch.no_grad():
            if torch.any(ten_initmask):
                x[ten_initmask, :] = x[ten_initmask, :] * 0  # to mask expressions kept for testing.
        xe = self.encoder_x(x)  # [N, dim_embedding]
        with torch.no_grad():
            if torch.any(ten_initmask):
                xe[ten_initmask, :] = xe[ten_initmask, :] * 0  # to mask expressions kept for testing.

        with torch.no_grad():
            pe = self._position_encoding(
                batch=batch,
                ten_xy_absolute=ten_xy_absolute
            ).detach()  # [N, dim_embedding]
            em_iscentralnode = self.embedding_iscentralnode(
                torch.tensor(
                    batch.batch_size * [1] + (x.size()[0] - batch.batch_size) * [0]
                ).to(ten_xy_absolute.device)
            ).detach()  # [N, 10]

            # define the masking token
            '''
            Note: Gene expression vectors are unknown in two cases
            - case 1. the expression vector is kept for testing.
            - case 2. the expression vector is known, but is kept hidden from the imputer.
                - done by setting to False with probability `prob_maskknowngenes`.
            In both cases the imputer must not be able to distinguish between the two.
            But the imputation loss is only defined on case 2.
            '''
            ten_masked_c1 = torch.tensor([False]*len(batch.y.tolist())).to(ten_xy_absolute.device)
              # no imputation now --> no unobserved expvect. (batch.y == MaskLabel.UNKNOWN_TEST.value).to(ten_xy_absolute.device)
            a = ~ten_masked_c1  # a; available expression vectors.
            a : torch.Tensor
            if torch.any(a) and (prob_maskknowngenes > 0.0):
                a[a == True] = a[a == True] & torch.tensor(
                    [np.random.rand() > prob_maskknowngenes for _ in range(torch.sum(a == True).tolist())]
                ).to(ten_xy_absolute.device)
            assert (torch.all(a))  # assert all expvects are available.
            em_blankorobserved = self.embedding_blankorobserved(
                a+0
            )  # [N, 10]
            ten_manually_masked = (~ten_masked_c1) & (~a)

            #mask xe
            if torch.any(~a):
                xe[~a, :] = xe[~a, :] * 0

        '''
        print("batch.x.shape = {}".format(batch.x.shape))
        print("batch.y.shape = {}".format(batch.y.shape))
        print("xe.shape = {}".format(xe.shape))
        print("pe.shape = {}".format(pe.shape))
        print("em_iscentralnode.shape = {}".format(em_iscentralnode.shape))
        print("em_blankorobserved.shape = {}".format(em_blankorobserved.shape))
        '''
        em_final = torch.cat(
            [xe+pe, em_iscentralnode, em_blankorobserved],
            1
        )

        return em_final, ten_manually_masked



class Disentangler(nn.Module):
    def __init__(self, maxsize_subgraph, kwargs_em1, kwargs_tformer1, str_mode_headxint_headxspl_headboth):
        '''
        :param maxsize_subgraph: the max size of the subgraph returned by pyg's NeighLoader.
        :param kwargs_em1:
        :param kwargs_tformer1: other than `channels` and `input_size` which is determined by `maxsize_subgraph`
        :param str_mode_headxint_headxspl_headboth:
            - headxint: means that only muxint is generated by the head, and `muxspl =  x_cnt - muxint`
            - headxspl: means that only muxspl is generated by the head, and `muxint =  x_cnt - muxspl`
            - headboth: means muxint and muxspl are create by two different heads, in which case p(x|x_int+x_spl) must be included in logp(.)
        '''
        super(Disentangler, self).__init__()
        self.str_mode_headxint_headxspl_headboth = str_mode_headxint_headxspl_headboth
        dim_tf1 = kwargs_em1['dim_embedding'] + kwargs_em1['dim_em_iscentralnode'] + kwargs_em1['dim_em_blankorobserved']
        self.module_em1 = SubgraphEmbeddingImpAndDisengl(**kwargs_em1)
        self.module_tf1 = Padder(
            Linformer(**{
                **{'input_size':maxsize_subgraph, 'channels':dim_tf1},
                **kwargs_tformer1
            })
        )

        if self.str_mode_headxint_headxspl_headboth == 'headxint':
            self.module_linearhead_muxint = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, dim_tf1),
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, kwargs_em1['num_genes'])
            )
            self.module_linearhead_muxspl = None
        elif self.str_mode_headxint_headxspl_headboth == 'headxspl':
            self.module_linearhead_muxspl = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, dim_tf1),
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, kwargs_em1['num_genes'])
            )
            self.module_linearhead_muxint = None
        elif self.str_mode_headxint_headxspl_headboth == 'headboth':
            self.module_linearhead_muxint = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, dim_tf1),
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, kwargs_em1['num_genes'])
            )
            self.module_linearhead_muxspl = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, dim_tf1),
                nn.LeakyReLU(),
                nn.Linear(dim_tf1, kwargs_em1['num_genes'])
            )
        else:
            raise Exception(
                "Uknown value {} for str_mode_headxint_headxspl_headboth.".format(self.str_mode_headxint_headxspl_headboth)
            )

        self.module_linearhead_sigmaxint = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tf1, kwargs_em1['num_genes'])
        )
        self.module_linearhead_sigmaxspl = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tf1, kwargs_em1['num_genes'])
        )
        self._check_args()

    def _check_args(self):
        assert (
            self.str_mode_headxint_headxspl_headboth in ['headxint', 'headxspl', 'headboth']
        )

    def forward(self, batch, prob_maskknowngenes:float, ten_xy_absolute:torch.Tensor):
        '''
        :param batch:
        :param prob_maskknowngenes: must be zero for `Disentangler`, this arg is kept for consistency.
        :param ten_xy_absolute:
        :return:
        '''
        assert(prob_maskknowngenes == 0.0)
        x_log1p = torch.log(
            1.0 + batch.x.to_dense()  # TODO: how to make sure that batch.x contains the count data ???
        ).to(ten_xy_absolute.device)
        x_cnt = batch.x.to_dense().to(ten_xy_absolute.device).detach() + 0.0
        ten_in_tf1, ten_manually_masked = self.module_em1(
            batch=batch,
            prob_maskknowngenes=0.0,
            ten_xy_absolute=ten_xy_absolute
        )  # [N, dim_tf1], [N]
        ten_out_tf1 = self.module_tf1(ten_in_tf1.unsqueeze(0))[0, :, :]  # [N, dim_tf1] in [-inf, inf]

        assert (not torch.any(ten_manually_masked))
        loss_imputex = None
        ten_out_imputer = 0.0



        # compute muxint, muxspl
        with torch.no_grad():
            oneon_x_nonzero = (x_cnt > 0.0) + 0.0  # [N, num_genes]

        if self.str_mode_headxint_headxspl_headboth == 'headxint':
            muxint = torch.clamp(
                self.module_linearhead_muxint(ten_out_tf1) * oneon_x_nonzero,
                min=0.00001 * torch.ones_like(x_cnt),  # TODO: maybe tune?
                max=x_cnt
            )  # [N, num_genes]
            muxspl = x_cnt - muxint  # [N, num_genes]
        elif self.str_mode_headxint_headxspl_headboth == 'headxspl':
            muxspl = torch.clamp(
                self.module_linearhead_muxspl(ten_out_tf1) * oneon_x_nonzero,
                min=0.00001 * torch.ones_like(x_cnt),  # TODO: maybe tune?
                max=x_cnt
            )  # [N, num_genes]
            muxint = x_cnt - muxspl  # [N, num_genes]
        elif self.str_mode_headxint_headxspl_headboth == 'headboth':
            muxint = torch.clamp(
                self.module_linearhead_muxint(ten_out_tf1) * oneon_x_nonzero,
                min=0.00001 * torch.ones_like(x_cnt),  # TODO: maybe tune?
                max=x_cnt
            )  # [N, num_genes]
            muxspl = torch.clamp(
                self.module_linearhead_muxspl(ten_out_tf1) * oneon_x_nonzero,
                min=0.00001 * torch.ones_like(x_cnt),  # TODO: maybe tune?
                max=x_cnt
            )  # [N, num_genes]
        else:
            raise Exception(
                "Uknown value {} for str_mode_headxint_headxspl_headboth.".format(self.str_mode_headxint_headxspl_headboth)
            )


        # compute sigmaxint, sigmaxspl
        sigmaxint = torch.clamp(
            self.module_linearhead_sigmaxint(ten_out_tf1),
            min=0.00001,  # TODO: maybe tune?
            max=torch.inf
        )  # [N, num_genes]
        sigmaxspl = torch.clamp(
            self.module_linearhead_sigmaxspl(ten_out_tf1),
            min=0.00001,  # TODO: maybe tune?
            max=torch.inf
        )  # [N, num_genes]
        return dict(
            muxint=muxint,
            muxspl=muxspl,
            sigmaxint=sigmaxint,
            sigmaxspl=sigmaxspl,
            ten_out_imputer=ten_out_imputer + 0.0,
            loss_imputex=loss_imputex
        )





