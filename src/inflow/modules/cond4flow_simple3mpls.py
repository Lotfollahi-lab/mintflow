
'''
The q(z,s | xbar_int, xbar_spl) with simple MLP heads.
'''

import numpy as np
import torch
import torch.nn as nn
from . import impanddisentgl


class Cond4FlowVarphi0SimpleMLPs(nn.Module):
    def __init__(self, dim_s:int, kwargs_genmodel, type_module_xbarint2z, kwargs_module_xbarint2z, type_module_xbarspl2sout, kwargs_module_xbarspl2sout, type_module_xbarspl2sin, kwargs_module_xbarspl2sin):
        '''
        :param dim_s: dimension of s (i.e. s-in s and s-out s).
        :param kwargs_genmodel: so this module knows wheter to add cell-type/niche labels for z and s_out.
        '''
        super(Cond4FlowVarphi0SimpleMLPs, self).__init__()
        # self.sigma_sz = sigma_sz
        self.kwargs_genmodel = kwargs_genmodel

        self.module_xbarint2z = type_module_xbarint2z(**kwargs_module_xbarint2z)  # when passed in, either operates on xbar_int or both [xbar_int, u_z]
        self.module_xbarspl2sout = type_module_xbarspl2sout(**kwargs_module_xbarspl2sout)  # when passed in, either operates on xbar_int or both [xbar_int, u_z]
        self.module_xbarspl2sin = type_module_xbarspl2sin(**kwargs_module_xbarspl2sin)  # when passed in, either operates on xbar_int or both [xbar_int, u_z]


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
        assert (batch.y.size()[1] == 2*num_celltypes)

        ten_uz = batch.y[:, 0:num_celltypes].to(ten_xy_absolute.device) if(self.kwargs_genmodel['flag_use_int_u']) else None
        ten_us = batch.y[:, num_celltypes::].to(ten_xy_absolute.device) if(self.kwargs_genmodel['flag_use_spl_u']) else None

        if ten_uz is None:
            mu_z = self.module_xbarint2z(ten_xbar_int)  # [N, dim_z]
        else:
            mu_z = self.module_xbarint2z(
                torch.cat([ten_xbar_int, ten_uz], 1)
            )  # [N, dim_z]


        if ten_us is None:
            mu_sout = self.module_xbarspl2sout(
                ten_xbar_spl
            )  # [N, dim_s]
        else:
            mu_sout = self.module_xbarspl2sout(
                torch.cat(
                    [ten_xbar_spl, ten_us],
                    1
                )
            )  # [N, dim_s]

        if ten_us is None:
            mu_sin = self.module_xbarspl2sin(
                ten_xbar_spl
            )  # [N, dim_s]
        else:
            mu_sin = self.module_xbarspl2sin(
                torch.cat(
                    [ten_xbar_spl, ten_us],
                    1
                )
            )  # [N, dim_s]



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
