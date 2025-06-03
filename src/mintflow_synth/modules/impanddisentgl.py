
import numpy as np
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from linformer_pytorch import Linformer, Padder


class MaskLabel(Enum):
    '''
    The mask-label of each cell is determined by `batch.y`, where batch is returned by pyg's NeirhborLoader.
    '''
    KNOWN = 0  # The cell's expression vector is known.
    UNKNOWN_TEST = 1  # The cell's expression vector is not shown to the trainer, and is to be used for testing phase.
    #UNKNOWN_LEARN_IMPUTATION = 1  # The cell's expression vector is really known, but is masked to train the imputer.

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
        ).unsqueeze(0) # [1, dby4]
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
        ten_initmask = (batch.y == MaskLabel.UNKNOWN_TEST.value)
        with torch.no_grad():
            x[ten_initmask, :] = x[ten_initmask, :] * 0  # to mask expressions kept for testing.
        xe = self.encoder_x(x)  # [N, dim_embedding]
        with torch.no_grad():
            xe[ten_initmask, :] = xe[ten_initmask, :] * 0  # to mask expressions kept for testing.

        with torch.no_grad():
            pe = self._position_encoding(
                batch=batch,
                ten_xy_absolute=ten_xy_absolute
            ).detach()  # [N, dim_embedding]
            em_iscentralnode = self.embedding_iscentralnode(
                torch.tensor(
                    batch.batch_size * [1] + (x.size()[0] - batch.batch_size) * [0]
                )
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
            ten_masked_c1 = (batch.y == MaskLabel.UNKNOWN_TEST.value)
            a = ~ten_masked_c1  # a; available expression vectors.
            a : torch.Tensor
            if torch.any(a==True) and (prob_maskknowngenes > 0.0):
                a[a == True] = a[a == True] & torch.tensor(
                    [np.random.rand() > prob_maskknowngenes for _ in range(torch.sum(a == True).tolist())]
                )
            em_blankorobserved = self.embedding_blankorobserved(
                a+0
            )  # [N, 10]
            ten_manually_masked = (~ten_masked_c1) & (~a)

            #mask xe
            xe[~a, :] = xe[~a, :] * 0

        em_final = torch.cat(
            [xe+pe, em_iscentralnode, em_blankorobserved],
            1
        )
        return em_final, ten_manually_masked


class ImputerAndDisentangler(nn.Module):
    def __init__(self, maxsize_subgraph, kwargs_em1, kwargs_em2, kwargs_tformer1,  kwargs_tformer2):
        '''
        :param maxsize_subgraph: the max size of the subgraph returned by pyg's NeighLoader.
        :param kwargs_em1:
        :param kwargs_em2:
        :param kwargs_tformer1: other than `channels` and `input_size` which is determined by `maxsize_subgraph`
        :param kwargs_tformer2: other than `channels` and `input_size` which is determined by `maxsize_subgraph`
        '''
        super(ImputerAndDisentangler, self).__init__()
        dim_tf1 = kwargs_em1['dim_embedding'] + kwargs_em1['dim_em_iscentralnode'] + kwargs_em1['dim_em_blankorobserved']
        #kwargs_em2['num_genes'] = dim_tf1
        self.module_em1 = SubgraphEmbeddingImpAndDisengl(**kwargs_em1)
        self.module_em2 = SubgraphEmbeddingImpAndDisengl(**kwargs_em2)
        self.module_tf1 = Padder(
            Linformer(**{
                **{'input_size':maxsize_subgraph, 'channels':dim_tf1},
                **kwargs_tformer1
            })
        )
        self.module_imputer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tf1, kwargs_em1['num_genes'])
        )
        dim_em2 = kwargs_em2['dim_embedding'] + kwargs_em2['dim_em_iscentralnode'] + kwargs_em2['dim_em_blankorobserved']
        dim_tf2 = dim_em2 + dim_tf1  # p.s. pg6
        self.dim_tf2 = dim_tf2
        self.module_tf2 = Padder(
            Linformer(**{
                **{'input_size':maxsize_subgraph, 'channels':dim_tf2},
                **kwargs_tformer2
            })
        )
        self.module_linearhead_muxint = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tf2, kwargs_em1['num_genes'])
        )  # there is no sigmaxspl, because that is computed by subtracting muxint from x_cnt.
        self.module_linearhead_sigmaxint = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tf2, kwargs_em1['num_genes'])
        )
        self.module_linearhead_sigmaxspl = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tf2, kwargs_em1['num_genes'])
        )


    def forward(self, batch, prob_maskknowngenes:float, ten_xy_absolute:torch.Tensor):
        '''
        :param batch:
        :param prob_maskknowngenes:
        :param ten_xy_absolute:
        :return:
        '''
        x_log1p = torch.log(
            1.0 + batch.x.to_dense()
        ).to(ten_xy_absolute.device)
        x_cnt = batch.x.to_dense().to(ten_xy_absolute.device) + 0.0
        ten_in_tf1, ten_manually_masked = self.module_em1(
            batch=batch,
            prob_maskknowngenes=prob_maskknowngenes,
            ten_xy_absolute=ten_xy_absolute
        )  # [N, dim_tf1], [N]
        ten_out_tf1 = self.module_tf1(ten_in_tf1.unsqueeze(0))[0,:,:]  # [N, dim_tf1] in [-inf, inf]
        loss_imputex = F.mse_loss(
            self.module_imputer(ten_out_tf1[ten_manually_masked, :]),
            x_log1p[ten_manually_masked, :]
        )  # TODO: probably change the loss ???
        ten_in_tf2_part1, _ = self.module_em2(
            batch=batch,
            prob_maskknowngenes=0.0,  # no manual masking for tf2, because the selfsup training has happened in tf1.
            ten_xy_absolute=ten_xy_absolute
        )  # [N, dim_em2],

        ten_out_tf2 = self.module_tf2(
            torch.cat([ten_in_tf2_part1, ten_out_tf1], 1).unsqueeze(0)
        )[0,:,:]  # [N, dim_tf2] in [-inf, inf]


        # compute muxint, muxspl, sigmaxint, sigmaxspl
        muxint = torch.clamp(
            self.module_linearhead_muxint(ten_out_tf2),
            min=0.00001*torch.ones_like(x_cnt),  # TODO: maybe tune?
            max=x_cnt
        )  # [N, num_genes]
        muxspl = x_cnt - muxint  # [N, num_genes]
        sigmaxint = torch.clamp(
            self.module_linearhead_sigmaxint(ten_out_tf2),
            min=0.00001,  # TODO: maybe tune?
            max=torch.inf
        )  # [N, num_genes]
        sigmaxspl = torch.clamp(
            self.module_linearhead_sigmaxspl(ten_out_tf2),
            min=0.00001,  # TODO: maybe tune?
            max=torch.inf
        )  # [N, num_genes]
        return dict(
            muxint=muxint,
            muxspl=muxspl,
            sigmaxint=sigmaxint,
            sigmaxspl=sigmaxspl,
            loss_imputex=loss_imputex
        )








