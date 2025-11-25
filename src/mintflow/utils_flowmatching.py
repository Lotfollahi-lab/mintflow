'''
Utilities for conditional flow matching
'''

from typing import Dict, Union, Callable, Tuple, Any, Optional
import torch
import time
import torchcfm
import numpy as np
from torch.nn.modules.module import T
from torch.utils.hooks import RemovableHandle
from torchcfm.optimal_transport import OTPlanSampler
from .modules import neuralODE
from enum import Enum
import torchdiffeq


class ModeSampleX0(Enum):
    RANDOM = 1  # X0 is not determined by inflow --> **Only** to be used in debug mode and for ablation study.
    FROMINFLOW = 2  # X0  (i.e. [Z0, S0] with the notation of inflow) comes from inflow itself.

class ModeMinibatchPerm(Enum):
    RANDOM = 1  # no matching
    OT = 2      # mini-batch OT

class ModeTimeSched(Enum):
    UNIFORM = 1 # uniform sampling in [0,1].


class ModeFMLoss(Enum):
    NOISEDIR = 1  # v(x_t, t) predicts 'x1 - x0', as done in torchcfm NBs.
    EDNPOINT = 2  # v(x_t, t) predicts `x1`, i.e. the endpoint parameterisation.

class FMLossCombinationMode(Enum):
    ZSIN_COMBINED = 1  # the FM loss is defined over [z, sin]
    ZSIN_SEPARATE = 2  # two FM losses are computed, one for [z], and one for [z, sin]

class ConditionalFlowMatcher:
    '''
    Wrapps all FM details (e.g. conditional sampling, mini-batch OT, etc.).
    '''
    def __init__(
            self, mode_samplex0:ModeSampleX0, mode_minibatchper:ModeMinibatchPerm,
            kwargs_otsampler:Dict, mode_timesched:ModeTimeSched, sigma:float, mode_fmloss:ModeFMLoss,
            fmloss_combinationmode:FMLossCombinationMode
        ):
        '''

        :param mode_samplex0:
        :param mode_minibatchper:
        :param kwargs_otsampler: the kwargs of `OTPlanSampler` like
            - method: 'sinkhorn', 'exact', etc.
            - reg: the regularization (the bigger the value --> less close to OT).
            - ...
        :param mode_timesched
        :param sigma: the added noise. It could depend on time as sigma_t, but for now we let it constant (as usually done).
        '''
        assert (
            isinstance(mode_samplex0, ModeSampleX0)
        )
        assert (
            isinstance(mode_minibatchper, ModeMinibatchPerm)
        )
        assert (
            isinstance(mode_timesched, ModeTimeSched)
        )
        assert (
            isinstance(mode_fmloss, ModeFMLoss)
        )
        assert (
            isinstance(fmloss_combinationmode, FMLossCombinationMode)
        )

        self.mode_samplex0 = mode_samplex0
        self.mode_minibatchper = mode_minibatchper
        self.kwargs_otsampler = kwargs_otsampler
        self.mode_timesched = mode_timesched
        self.sigma = sigma
        self.mode_fmloss = mode_fmloss
        self.fmloss_combinationmode = fmloss_combinationmode


        if self.mode_minibatchper == ModeMinibatchPerm.OT:
            self.ot_sampler = OTPlanSampler(**self.kwargs_otsampler)



    @torch.no_grad()
    def _sample_x0(self, x1, x0_frominflow):
        if self.mode_samplex0 == ModeSampleX0.RANDOM:
            return torch.randn_like(x1)
        elif self.mode_samplex0 == ModeSampleX0.FROMINFLOW:
            return x0_frominflow
        else:
            raise NotImplementedError("ddd")



    @torch.no_grad()
    def _perm_batches(self, x0:torch.Tensor, x1:torch.Tensor):

        assert x0.size()[1] % 2 == 0
        assert len(x0.size()) == 2
        D = x0.size()[1] // 2

        if self.mode_minibatchper == ModeMinibatchPerm.RANDOM:

            if self.fmloss_combinationmode == FMLossCombinationMode.ZSIN_COMBINED:
                return {
                    "ZSin":{'x0':x0, 'x1':x1, 'time_taken':0.0}
                }
            elif self.fmloss_combinationmode == FMLossCombinationMode.ZSIN_SEPARATE:
                return {
                    "Z":{'x0':x0[:, 0:D], 'x1':x1[:, 0:D], 'time_taken':0.0},
                    "ZSin":{'x0':x0, 'x1':x1, 'time_taken':0.0}
                }
            else:
                raise Exception(
                    "Unknown `fmloss_combinationmode` {}".format(self.fmloss_combinationmode)
                )

        elif self.mode_minibatchper == ModeMinibatchPerm.OT:

            if self.fmloss_combinationmode == FMLossCombinationMode.ZSIN_COMBINED:
                t_tic = time.time()
                
                ot_map = self.ot_sampler.get_map(x0, x1)
                list_0_ot_1 = [np.where(ot_map[i, :] > 0)[0].tolist()[0] for i in range(ot_map.shape[0])]

                new_x0 = x0 + 0.0
                new_x1 = x1[list_0_ot_1, :] + 0.0

                time_taken = time.time() - t_tic

                return {"ZSin":{'x0':new_x0, 'x1':new_x1, 'time_taken':time_taken}}
            
            elif self.fmloss_combinationmode == FMLossCombinationMode.ZSIN_SEPARATE:
                # onece for ZSin combined
                t_tic = time.time()
                
                ot_map = self.ot_sampler.get_map(x0, x1)
                list_0_ot_1 = [np.where(ot_map[i, :] > 0)[0].tolist()[0] for i in range(ot_map.shape[0])]

                time_taken = time.time() - t_tic

                dict_toret = {"ZSin":{'x0':x0 + 0.0, 'x1':x1[list_0_ot_1, :] + 0.0, 'time_taken':time_taken + 0.0}}

                # now separately for Z
                t_tic = time.time()
                
                ot_map = self.ot_sampler.get_map(x0[:, 0:D], x1[:, 0:D])
                list_0_ot_1 = [np.where(ot_map[i, :] > 0)[0].tolist()[0] for i in range(ot_map.shape[0])]

                time_taken = time.time() - t_tic

                dict_toret['Z'] = {'x0':x0[:, 0:D] + 0.0, 'x1':x1[list_0_ot_1, :][:, 0:D] + 0.0, 'time_taken':time_taken + 0.0}

                return dict_toret


            else:
                raise Exception(
                    "Unknown `fmloss_combinationmode` {}".format(self.fmloss_combinationmode)
                )
                
            
        else:
            raise NotImplementedError("ddd")


    @torch.no_grad()
    def _gen_t(self, batch_size):
        if self.mode_timesched == ModeTimeSched.UNIFORM:
            return torch.rand(size=[batch_size])
        else:
            raise NotImplementedError("ddd")

    def _get_fmloss(self, module_v:neuralODE.MLP, x0, x1, xt, t, ten_batchEmb):
        # make ut
        if self.mode_fmloss == ModeFMLoss.NOISEDIR:
            ut = x1 - x0
        elif self.mode_fmloss == ModeFMLoss.EDNPOINT:
            ut = x1
        else:
            raise NotImplementedError("ddd")

        '''
        vt = module_v(
            torch.cat(
                [ten_batchEmb, xt, t.flatten()[:, None]],
                dim=-1
            )
        )
        '''
        vt = module_v(
            t=t,
            x=xt,
            ten_BatchEmb=ten_batchEmb
        )
        return torch.mean(
            torch.sum(
                (vt - ut) ** 2,
                1
            )
        )

    def get_fmloss(
        self,
        module_v:neuralODE.MLP,
        x1:torch.Tensor,
        x0_frominflow:torch.Tensor,
        ten_batchEmb:torch.Tensor
    ):
        '''
        :param module_v: the V(.) module.
        :param x1: a mini-batch of samples from p_1(.).
        :param x0_frominflow: inflow decides x0, but **only** for ablation study x0 could be generated compltely randomly.
        :param ten_batchEmb: the batch embeddings (as stored in the pyg batch).
        :return:
        '''
        assert (isinstance(module_v, neuralODE.MLP))

        # sample x0 and compute the permutation (s)
        with torch.no_grad():  # TODO: should it be here? The sample notebooks don't put no_grad().
            x0 = self._sample_x0(
                x1=x1,
                x0_frominflow=x0_frominflow
            ) # [N, D].
            # x0, x1, time_taken = self._perm_batches(x0, x1)  # [N, D], [N, D]

            dict_vectname_to_dict_x0x1timetaken = self._perm_batches(x0, x1)

        dict_vectname_to_dict_fmloss_timetaken = {}

        for vectname in dict_vectname_to_dict_x0x1timetaken.keys():
            x0 = dict_vectname_to_dict_x0x1timetaken[vectname]['x0']
            x1 = dict_vectname_to_dict_x0x1timetaken[vectname]['x1']
            time_taken = dict_vectname_to_dict_x0x1timetaken[vectname]['time_taken']

            with torch.no_grad():  # TODO: is no_grad() needed here? 
                t = self._gen_t(batch_size=x1.size()[0]).unsqueeze(-1).to(x1.device)  # [N, 1]
                xt = t * x1 + (1 - t) * x0 + self.sigma * torch.randn_like(x1)

            

            dict_vectname_to_dict_fmloss_timetaken[vectname] = {
                'fmloss':self._get_fmloss(
                    module_v=module_v,
                    x0=x0,
                    x1=x1,
                    xt=xt,
                    t=t[:,0],
                    ten_batchEmb=ten_batchEmb
                ),
                'time_taken':time_taken
            }


        return dict_vectname_to_dict_fmloss_timetaken    








class WrapperTorchDiffEq(torch.nn.Module):
    def __init__(self, model:neuralODE.MLP, kwargs_odeint:dict):
        super(WrapperTorchDiffEq, self).__init__()
        assert isinstance(model, neuralODE.MLP)
        self.model = model
        self.kwargs_odeint = kwargs_odeint

    def forward(self, t_in, x_in, ten_BatchEmb_in):

        '''
        print("args passed to `WrapperTorchDiffEq`")
        print("   t_in.shape = {}".format(t_in.shape))
        print("   x_in.shape = {}".format(x_in.shape))
        print("   ten_BatchEmb_in.shape = {}".format(ten_BatchEmb_in.shape))
        print("==========\n\n\n")

        args passed to `WrapperTorchDiffEq`
           t_in.shape = torch.Size([10])
           x_in.shape = torch.Size([13, 200])
           ten_BatchEmb_in.shape = torch.Size([13, 4])
        ==========
        '''

        traj = torchdiffeq.odeint(
            lambda t, x: self.model.forward_4torchdiffeq(t, x, ten_BatchEmb_in),
            y0=x_in,
            t=t_in,
            **self.kwargs_odeint
        )

        '''
        print("traj.shape = {}".format(traj.shape))

        traj.shape = torch.Size([10, 351, 200])  Note: with `tnum_steps` eqaul to 10.
        '''

        return traj[-1, :, :]  # [batchsize x dim_z+dim_s]










