'''
Utilities for conditional flow matching
'''

from typing import Dict
import torch
import torchcfm
from torchcfm.optimal_transport import OTPlanSampler
from .modules import neuralODE
from enum import Enum


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

class ConditionalFlowMatcher:
    '''
    Wrapps all FM details (e.g. conditional sampling, mini-batch OT, etc.).
    '''
    def __init__(self, mode_samplex0:ModeSampleX0, mode_minibatchper:ModeMinibatchPerm, kwargs_otsampler:Dict, mode_timesched:ModeTimeSched, sigma:float, mode_fmloss:ModeFMLoss):
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
        self.mode_samplex0 = mode_samplex0
        self.mode_minibatchper = mode_minibatchper
        self.kwargs_otsampler = kwargs_otsampler
        self.mode_timesched = mode_timesched
        self.sigma = sigma
        self.mode_fmloss = mode_fmloss


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
        if self.mode_minibatchper == ModeMinibatchPerm.RANDOM:
            return x0, x1
        elif self.mode_minibatchper == ModeMinibatchPerm.OT:
            x0, x1 = self.ot_sampler.sample_plan(x0, x1)
            return x0, x1
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


        vt = module_v(
            torch.cat(
                [ten_batchEmb, xt, t.flatten()[:, None]],
                dim=-1
            )
        )
        return torch.mean((vt - ut) ** 2)

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

        # sample xt
        with torch.no_grad():  # TODO: should it be here? The sample notebooks don't put no_grad().
            x0 = self._sample_x0(
                x1=x1,
                x0_frominflow=x0_frominflow
            ) # [N, D].
            x0, x1 = self._perm_batches(x0, x1)  # [N, D], [N, D]
            t = self._gen_t(batch_size=x1.size()[0]).unsqueeze(-1).to(x1.device)  # [N, 1]
            xt = t * x1 + (1 - t) * x0 + self.sigma * torch.randn_like(x1)

        # return loss
        return self._get_fmloss(
            module_v=module_v,
            x0=x0,
            x1=x1,
            xt=xt,
            t=t,
            ten_batchEmb=ten_batchEmb
        )








