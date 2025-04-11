
'''
The distirbution \varphi_{flow0} in the paper (i.e. the condition for flow matching).
'''

import numpy as np
import torch
import torch.nn as nn
from linformer_pytorch import Linformer, Padder
from . import impanddisentgl


class SubgraphEmbeddingCond4Flow(nn.Module):
    def __init__(self, dim_x, dim_embedding, dim_em_iscentralnode, dim_em_blankorobserved):
        '''
        :param dim_x: dimension of input (i.e. before pe is added).
        :param dim_embedding: The dim of embedding for each cell, must be a multiple of 4.
        '''
        super(SubgraphEmbeddingCond4Flow, self).__init__()
        # grab args
        self.dim_embedding = dim_embedding
        self.dim_em_iscentralnode = dim_em_iscentralnode
        self.dim_em_blankorobserved = dim_em_blankorobserved
        assert(self.dim_embedding%4 == 0)
        # make internals
        self.encoder_x_spl = nn.Linear(
            dim_x,
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
    def _position_encoding(self, batch, ten_xy_absolute: torch.Tensor):
        '''
        The positional embedding as done in cellplm paper.
        :return:
        '''
        ten_xy = ten_xy_absolute[batch.n_id.tolist(), :] + 0.0  # [N, 2] where N is the size of subgraph.

        # min-max normaliztion to get xy in [0, 100]
        ten_xy = ten_xy - torch.min(ten_xy, 0).values.unsqueeze(0)  # [N, 2]
        ten_xy = ten_xy / torch.clamp(torch.max(ten_xy, 0).values.unsqueeze(0), min=0.0001, max=torch.inf)  # [N, 2]
        ten_xy = 100.0 * ten_xy  # [N,2] in [0,100]

        #  compute pe
        dby4 = self.dim_embedding // 4
        denum = torch.exp(
            (torch.tensor(range(dby4)) / dby4) * np.log(10000.0)
        ).unsqueeze(0).to(ten_xy_absolute.device)  # [1, dby4]
        x_sin = torch.sin(ten_xy[:, 0].unsqueeze(1) / denum)  # [N, bdy4]
        x_cos = torch.cos(ten_xy[:, 0].unsqueeze(1) / denum)  # [N, bdy4]
        y_sin = torch.sin(ten_xy[:, 1].unsqueeze(1) / denum)  # [N, bdy4]
        y_cos = torch.cos(ten_xy[:, 1].unsqueeze(1) / denum)  # [N, bdy4]
        pe = torch.cat(
            [x_sin, x_cos, y_sin, y_cos],
            1
        )  # [N, self.dim_embedding]
        return pe

    def forward(self, batch, ten_xbar_spl, ten_xy_absolute: torch.Tensor):
        '''
        :param batch: the batch returned by pyg's neighborloader.
            Only batch.y is used here.
        :param ten_xbar_spl: the predicted x_spl, i.e. the output from encoder to diffusion latet space.
            The range is left as whatever tf2 sends out, which is cnt matrix rather than log1p.
        :param ten_xy_absolute: the xy coordinates for the entire graph (not only the batch).
        :return:
        '''
        assert (ten_xy_absolute.size()[0] > ten_xbar_spl.size()[0])
        x = ten_xbar_spl.to(ten_xy_absolute.device)
        ten_initmask = torch.tensor([False] * x.size()[0])
        # (batch.y == impanddisentgl.MaskLabel.UNKNOWN_TEST.value)


        xe = self.encoder_x_spl(x)  # [N, dim_embedding]


        with (torch.no_grad()):
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



            ten_masked_c1 = torch.tensor([False] * x.size()[0]).to(ten_xy_absolute.device)
            # (batch.y == impanddisentgl.MaskLabel.UNKNOWN_TEST.value).to(ten_xy_absolute.device)
            a = ~ten_masked_c1  # a; available expression vectors.
            em_blankorobserved = self.embedding_blankorobserved(
                a + 0
            )  # [N, 10]



        em_final = torch.cat(
            [xe + pe, em_iscentralnode, em_blankorobserved],
            1
        )
        return em_final




class Cond4FlowVarphi0(nn.Module):
    def __init__(self, maxsize_subgraph:int, dim_s:int, kwargs_em_spl,
                 kwargs_tformer_spl, type_module_xbartoz, kwargs_module_xbartoz, kwargs_genmodel):
        '''
        :param maxsize_subgraph: the max size of the subgraph returned by pyg's NeighLoader.
        :param kwargs_em_spl:
        :param dim_s: dimension of s (i.e. s-in s and s-out s).
        :param kwargs_tformer_spl: other than `channels` and `input_size` which is determined by `maxsize_subgraph`
        :param type_module_xbartoz: type of the module that maps xbar_int to z. Can be, e.g., a MLP.
        :param kwargs_module_xbartoz: kwargs of the module that maps xbar_int to z.
        :param kwargs_genmodel: so this module knows wheter to add cell-type/niche labels for z and s_out.
        '''
        super(Cond4FlowVarphi0, self).__init__()
        # self.sigma_sz = sigma_sz
        self.kwargs_genmodel = kwargs_genmodel
        dim_tfspl = kwargs_em_spl['dim_embedding'] +\
                    kwargs_em_spl['dim_em_iscentralnode'] + kwargs_em_spl['dim_em_blankorobserved']
        # TODO: dim_tfspl is same as dim_s in the genrative model. It has to be somewhere checked.
        self.module_em_spl = SubgraphEmbeddingCond4Flow(**kwargs_em_spl)
        self.module_tf_spl_sin = Padder(
            Linformer(**{
                **{'input_size':maxsize_subgraph, 'channels':dim_tfspl},
                **kwargs_tformer_spl
            })
        )
        self.module_tf_spl_sout = Padder(
            Linformer(**{
                **{'input_size': maxsize_subgraph, 'channels': dim_tfspl},
                **kwargs_tformer_spl
            })
        )
        self.module_xbarint_to_z = type_module_xbartoz(**kwargs_module_xbartoz)  # when passed in, either operates on xbar_int or both [xbar_int, u_z]

        # # infer dim_z
        # with torch.no_grad():
        #     dim_z = self.module_xbarint_to_z(torch.randn(1, kwargs_em_spl['dim_x'])).size()[1]

        # for linear head mu_s to be placed atop tf_spl
        num_celltypes = self.kwargs_genmodel['dict_varname_to_dim']['cell-types']
        self.module_head_mus_sin = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tfspl, dim_s)
        )  # TODO: double-check if it should be conditioned on u_s
        self.module_head_mus_sout = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(dim_tfspl + (self.kwargs_genmodel['dict_varname_to_dim']['u_spl'] if(self.kwargs_genmodel['flag_use_spl_u']) else 0), dim_s)
        )  # TODO: double-check the way of conditioning on u_s

    def forward(self, ten_xbar_int, batch, ten_xbar_spl, ten_xy_absolute: torch.Tensor):
        '''
        :param ten_xbar_int:
        :param batch: only used for position encoding (batch.x is not used).
        :param ten_xbar_spl:
        :param ten_xy_absolute:
        :return:
        '''
        # get u_z and u_s_out
        num_celltypes = self.kwargs_genmodel['dict_varname_to_dim']['cell-types']

        assert (
            batch.y.size()[1] == (batch.MINTFLOWMETAINF['dim_u_int'] + batch.MINTFLOWMETAINF['dim_u_spl'])
        )
        ten_uz = batch.y[:, 0:batch.MINTFLOWMETAINF['dim_u_int']].to(ten_xy_absolute.device) if (self.kwargs_genmodel['flag_use_int_u']) else None
        ten_us = batch.y[:, batch.MINTFLOWMETAINF['dim_u_int']::].to(ten_xy_absolute.device) if (self.kwargs_genmodel['flag_use_spl_u']) else None


        in_tf = self.module_em_spl(
            batch=batch,
            ten_xbar_spl=ten_xbar_spl,
            ten_xy_absolute=ten_xy_absolute
        )  # [N, dim_tfspl]
        mu_sin = self.module_head_mus_sin(
            self.module_tf_spl_sin(in_tf.unsqueeze(0))[0,:,:]
        )  # [N, dim_s]

        if ten_us is None:
            mu_sout = self.module_head_mus_sout(
                self.module_tf_spl_sout(in_tf.unsqueeze(0))[0, :, :]
            )  # [N, dim_s]
        else:
            mu_sout = self.module_head_mus_sout(
                torch.cat(
                    [self.module_tf_spl_sout(in_tf.unsqueeze(0))[0, :, :], ten_us],
                    1
                )
            )  # [N, dim_s]

        if ten_uz is None:
            mu_z = self.module_xbarint_to_z(ten_xbar_int)  # [N, dim_z]
        else:
            mu_z = self.module_xbarint_to_z(
                torch.cat([ten_xbar_int, ten_uz], 1)
            )  # [N, dim_z]


        sigma_sz = torch.ones(
            size=[mu_z.size()[1]+mu_sin.size()[1]],
            device=mu_sin.device
        )  # TODO: change it to learnable but with lower-bound clipping.
        return dict(
            mu_z=mu_z,
            mu_sin=mu_sin,
            mu_sout=mu_sout,
            sigma_sz=sigma_sz
        )











