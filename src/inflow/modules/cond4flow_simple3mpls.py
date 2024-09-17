
'''
The q(z,s | xbar_int, xbar_spl) with simple MLP heads.
'''

import numpy as np
import torch
import torch.nn as nn
from . import impanddisentgl
from . import mlp


class Cond4FlowVarphi0SimpleMLPs(nn.Module):
    def __init__(self, kwargs_genmodel, encZ_list_dim_hidden, encSin_list_dim_hidden, encSout_list_dim_hidden, dict_varname_to_takeCT_takeNCC, flag_use_layernorm:bool):
        '''
        :param kwargs_genmodel: so this module knows wheter to add cell-type/niche labels for z and s_out.
        :param encZ_list_dim_hidden: list of hidden dimenions of the MLP encoder for Z (in dim and out dim are automatically determined).
        :param dict_varname_to_takeCT_takeNCC: a dictionary with keys CT and NCC and values [take_CT:bool, take_NCC:bool]
        '''
        super(Cond4FlowVarphi0SimpleMLPs, self).__init__()
        self.kwargs_genmodel = kwargs_genmodel
        self.encZ_list_dim_hidden = encZ_list_dim_hidden
        self.encSin_list_dim_hidden = encSin_list_dim_hidden
        self.encSout_list_dim_hidden = encSout_list_dim_hidden
        self.dict_varname_to_takeCT_takeNCC = dict_varname_to_takeCT_takeNCC
        self.flag_use_layernorm = flag_use_layernorm

        self._check_args()

        dim_s = self.kwargs_genmodel['dict_varname_to_dim']['s']

        #create encoder modules
        self.module_enc_z = mlp.SimpleMLP(
            dim_input=dim_s + self._func_varname_to_dimextention('z'),
            list_dim_hidden=encZ_list_dim_hidden,
            dim_output=dim_s,
            bias=True,
            flag_endwithReLU=False,  # TODO:check
            flag_startswithReLU=False,  # TODO:check
            flag_use_layernorm=self.flag_use_layernorm
        )
        self.module_enc_sin = mlp.SimpleMLP(
            dim_input=dim_s + self._func_varname_to_dimextention('sin'),
            list_dim_hidden=encSin_list_dim_hidden,
            dim_output=dim_s,
            bias=True,
            flag_endwithReLU=False,  # TODO:check
            flag_startswithReLU=False,  # TODO:check
            flag_use_layernorm=self.flag_use_layernorm
        )
        self.module_enc_sout = mlp.SimpleMLP(
            dim_input=2*dim_s + self._func_varname_to_dimextention('sout'),
            list_dim_hidden=encSout_list_dim_hidden,
            dim_output=dim_s,
            bias=True,
            flag_endwithReLU=False,  # TODO:check
            flag_startswithReLU=False,  # TODO:check
            flag_use_layernorm=self.flag_use_layernorm
        )


    @torch.no_grad()
    def _func_extendby_CTNCC(self, x, batch, varname):
        '''
        Give a batch, returns the CT and NCC extension (if specified).
        The output is to be concatenated to the encoder's input.
        :param batch:
        :return:
        '''
        assert (varname in ['z', 'sin', 'sout'])
        output = [x]

        if self.dict_varname_to_takeCT_takeNCC[varname][0]:
            rng_CT = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
            ]
            output.append(
                batch.y[
                    :,
                    rng_CT[0]:rng_CT[1]
                ].to(x.device)
            )

        if self.dict_varname_to_takeCT_takeNCC[varname][1]:
            rng_NCC = batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
            output.append(
                batch.y[
                    :,
                    rng_NCC:
                ].to(x.device)
            )

        if len(output) > 1:
            return torch.cat(output, 1)
        else:
            return output[0]


    def _func_varname_to_dimextention(self, varname):
        assert (varname in ['z', 'sin', 'sout'])
        dim_toret = 0

        if self.dict_varname_to_takeCT_takeNCC[varname][0]:
            dim_toret += self.kwargs_genmodel['dict_varname_to_dim']['CT']

        if self.dict_varname_to_takeCT_takeNCC[varname][1]:
            dim_toret += self.kwargs_genmodel['dict_varname_to_dim']['NCC']

        return dim_toret



    def _check_args(self):
        assert (
            self.flag_use_layernorm in [True, False]
        )
        assert (
            set(self.dict_varname_to_takeCT_takeNCC.keys()) == {'z', 'sin', 'sout'}
        )
        for varname in ['z', 'sin', 'sout']:
            assert isinstance(self.dict_varname_to_takeCT_takeNCC[varname], list)
            assert (self.dict_varname_to_takeCT_takeNCC[varname][0] in [True, False])
            assert (self.dict_varname_to_takeCT_takeNCC[varname][1] in [True, False])

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
            batch.y.size()[1] == (batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'])
        )


        mu_z = self.module_enc_z(
            self._func_extendby_CTNCC(
                x=ten_xbar_int,
                batch=batch,
                varname='z'
            )
        )
        mu_sin = self.module_enc_sin(
            self._func_extendby_CTNCC(
                x=ten_xbar_spl,
                batch=batch,
                varname='sin'
            )
        )
        mu_sout = self.module_enc_sout(
            self._func_extendby_CTNCC(
                x=torch.cat([ten_xbar_int, ten_xbar_spl], 1),
                batch=batch,
                varname='sout'
            )
        )


        sigma_sz = 0.0001*torch.ones(
            size=[mu_z.size()[1]+mu_sin.size()[1]],
            device=mu_sin.device
        )  # TODO: change it to learnable but with lower-bound clipping.
        # now `sigma_sz` is ignored (manually sigma values are used instead).

        return dict(
            mu_z=mu_z,
            mu_sin=mu_sin,
            mu_sout=mu_sout,
            sigma_sz=sigma_sz
        )
