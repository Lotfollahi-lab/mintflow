

import torch
import numpy as np
from typing import List
import torch.nn as nn
from torch.distributions.normal import Normal
from torchdyn.core import NeuralODE
from torchcfm.utils import torch_wrapper
import torch_geometric as pyg
from scvi.distributions import ZeroInflatedNegativeBinomial
from . import utils
from . import probutils
from .modules import mlp
from . import zs_samplers
from . import utils_flowmatching



class DummyModule(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(DummyModule, self).__init__()
    def forward(self, x):
        return x

class MintFlowGenerativeModel(nn.Module):
    def __init__(
            self,
            num_cells,
            dict_varname_to_dim,
            dict_pname_to_scaleandunweighted,
            type_theta_aggr, kwargs_theta_aggr,
            type_moduleflow, kwargs_moduleflow,
            type_w_dec, kwargs_w_dec,
            kwargs_negbin_int, kwargs_negbin_spl,
            initval_thetanegbin_int:float | str, flag_train_negbintheta_int:bool, negbintheta_int_clamp_minmax:List[float] | None,
            initval_thetanegbin_spl:float | str, flag_train_negbintheta_spl:bool, negbintheta_spl_clamp_minmax:List[float] | None,
            flag_use_int_u: bool, module_int_mu_u: nn.Module | None, module_int_cov_u: mlp.SimpleMLPandExp | None,
            flag_use_spl_u: bool, module_spl_mu_u: nn.Module | None, module_spl_cov_u: mlp.SimpleMLPandExp | None,
            coef_zinb_int_loglik: float,
            coef_zinb_spl_loglik: float,
            dict_config_batchtoken: dict
    ):
        '''

        :param dict_varname_to_dim: a dict with keys
            - s
            - z
            - x
            - TODO:maybe more
        :param dict_pname_to_scaleandunweighted, with keys
            - z
            - sin
            - sout
            - xbar_int
            - xbar_spl
            - x: If set to [None, None] it means that p(x | xbar_int, xbar_spl) is not consider because, e.g., the decoder enforces that by design.
                Otherwise it contains [scale, flag_unweighted] as usual.
        :param initval_thetanegbin_int:float | str, flag_train_negbintheta_int:bool, negbintheta_clamp_minmax:List[int] | None: the negbin theta parameters for intrinsic.
            `initval_thetanegbin_int` could be either a float, or 'rand'.
            `flag_train_negbintheta_int` if set to True, the negbin theta parameter is trained.
            `negbintheta_clamp_minmax`: either a list of floats of length 2 (the min/max clip values for theta) or None, which means no clipping is applied.

        :param initval_thetanegbin_spl:float | str, flag_train_negbintheta_spl:bool, negbintheta_clamp_minmax:List[float] | None: the negbin theta parameters for intrinsic.
            `initval_thetanegbin_int` could be either a float, or 'rand'.
            `flag_train_negbintheta_int` if set to True, the negbin theta parameter is trained.
            `negbintheta_int_clamp_minmax`: either a list of floats of length 2 (the min/max clip values for theta) or None, which means no clipping is applied.
        :param module_z_sampler: "only" to be used when generating data (in the `sample` function), and not used when computing the loglik.
        :param module_s_sampler: "only" to be used when generating data (in the `sample` function), and not used when computing the loglik.
        :param flag_use_int_u, module_int_mu_u, module_int_cov_u: whether the u-label (with the notation of iVAE) is used for z, and modules to produce
            the mean and cov of p(z|u).
        :param flag_use_spl_u, module_spl_mu_u, module_spl_cov_u: whether the u-label (with the notation of iVAE) is used for s_out, and modules to produce
            the mean and cov of p(s_out|u).
        :dict_config_batchtoken: the configdict for how/if batchtoken is used. Containing keys
            - flag_enable_batchtoken_flowmodule: whether the flow part of decoder is conditioned on batch token.
            - TODO: maybe add more?
        :param TODO:complete


        '''
        super(MintFlowGenerativeModel, self).__init__()
        #grab args ===
        self.num_cells = num_cells
        self.dict_varname_to_dim = dict_varname_to_dim
        # self.dict_sigma2s = dict_sigma2s
        self.dict_pname_to_scaleandunweighted = dict_pname_to_scaleandunweighted
        # self.module_z_sampler = module_z_sampler
        # self.module_s_sampler = module_s_sampler
        self.kwargs_negbin_int, self.kwargs_negbin_spl = kwargs_negbin_int, kwargs_negbin_spl
        self.initval_thetanegbin_int, self.initval_thetanegbin_spl = initval_thetanegbin_int, initval_thetanegbin_spl
        self.flag_train_negbintheta_int, self.flag_train_negbintheta_spl = flag_train_negbintheta_int, flag_train_negbintheta_spl
        self.negbintheta_int_clamp_minmax, self.negbintheta_spl_clamp_minmax = negbintheta_int_clamp_minmax, negbintheta_spl_clamp_minmax
        self.flag_use_int_u, self.module_int_mu_u, self.module_int_cov_u = flag_use_int_u, module_int_mu_u, module_int_cov_u
        self.flag_use_spl_u, self.module_spl_mu_u, self.module_spl_cov_u = flag_use_spl_u, module_spl_mu_u, module_spl_cov_u
        self.coef_zinb_int_loglik = coef_zinb_int_loglik
        self.coef_zinb_spl_loglik = coef_zinb_spl_loglik
        self.dict_config_batchtoken = dict_config_batchtoken

        assert self.coef_zinb_int_loglik == 1.0, "coef_zinb_int_loglik has to be 1.0, since that's handled by annealing now."
        assert self.coef_zinb_spl_loglik == 1.0, "coef_zinb_spl_loglik has to be 1.0, since that's handled by annealing now."


        #make internals ===
        self.module_theta_aggr = type_theta_aggr(
            **{**{
                'dim_input':self.dict_varname_to_dim['s'],
                'dim_output':self.dict_varname_to_dim['s']},
                **kwargs_theta_aggr
            }
        ) #TODO: add note about requiring the `dim_input` and `dim_output` arguments.
        # TODO: `dim_input` and `dim_output` arguments were removed after adding batch token. Any issues?
        self.module_Vflow_unwrapped = type_moduleflow(
            **{**{
                'flag_enable_batchtoken_flowmodule':self.dict_config_batchtoken['flag_enable_batchtoken_flowmodule'],
                'dim_b':self.dict_varname_to_dim['BatchEmb'],
                'dim_z': self.dict_varname_to_dim['z'],
                'dim_s': self.dict_varname_to_dim['s']},
               **kwargs_moduleflow
            }
        )

        self.module_flow = utils_flowmatching.WrapperTorchDiffEq(
            model=self.module_Vflow_unwrapped,
            kwargs_odeint={
                'atol':1e-4,
                'rtol':1e-4,
                'method':"dopri5"
            }
        )
        # torch_wrapper is not needed:
        # https://github.com/atong01/conditional-flow-matching/blob/62c44affd877a01b7838d408b5dc4cbcbf83e3ad/examples/images/conditional_mnist.ipynb
        self.module_w_dec_int = type_w_dec(
            **{**{
                'dim_input': self.dict_varname_to_dim['BatchEmb'] + self.dict_varname_to_dim['z'],
                'dim_output': self.dict_varname_to_dim['x']},
               **kwargs_w_dec
               }
        )

        # self.module_w_dec_spl = self.module_w_dec_int
        self.module_w_dec_spl = type_w_dec(
            **{**{
                'dim_input': self.dict_varname_to_dim['BatchEmb'] + self.dict_varname_to_dim['z'],
                'dim_output': self.dict_varname_to_dim['x']},
               **kwargs_w_dec
               }
        )

        # make the theta parameters of NegBin intrinsic and NegBin spatial
        # for intrinsic
        if not isinstance(self.initval_thetanegbin_int, str):
            self.theta_negbin_int = torch.nn.Parameter(
                self.initval_thetanegbin_int * torch.ones(self.dict_varname_to_dim['x']).unsqueeze(0),
                requires_grad=self.flag_train_negbintheta_int
            )
        else:
            assert (self.initval_thetanegbin_int == 'rand')
            self.theta_negbin_int = torch.nn.Parameter(
                torch.empty(size=[self.dict_varname_to_dim['x']]).unsqueeze(0),
                requires_grad=self.flag_train_negbintheta_int
            )
        with torch.no_grad():
            if not (self.negbintheta_int_clamp_minmax is None):
                self.theta_negbin_int.clamp_(
                    self.negbintheta_int_clamp_minmax[0],
                    self.negbintheta_int_clamp_minmax[1]
                )

        # for spatial
        if not isinstance(self.initval_thetanegbin_spl, str):
            self.theta_negbin_spl = torch.nn.Parameter(
                self.initval_thetanegbin_spl * torch.ones(self.dict_varname_to_dim['x']).unsqueeze(0),
                requires_grad=self.flag_train_negbintheta_spl
            )
        else:
            assert (self.initval_thetanegbin_spl == 'rand')
            self.theta_negbin_spl = torch.nn.Parameter(
                torch.empty(size=[self.dict_varname_to_dim['x']]).unsqueeze(0),
                requires_grad=self.flag_train_negbintheta_spl
            )
        with torch.no_grad():
            if not (self.negbintheta_spl_clamp_minmax is None):
                self.theta_negbin_spl.clamp_(
                    self.negbintheta_spl_clamp_minmax[0],
                    self.negbintheta_spl_clamp_minmax[1]
                )



        self._check_args()


    def clamp_thetanegbins(self):
        with torch.no_grad():
            if not (self.negbintheta_int_clamp_minmax is None):
                self.theta_negbin_int.clamp_(
                    self.negbintheta_int_clamp_minmax[0],
                    self.negbintheta_int_clamp_minmax[1]
                )

        with torch.no_grad():
            if not (self.negbintheta_spl_clamp_minmax is None):
                self.theta_negbin_spl.clamp_(
                    self.negbintheta_spl_clamp_minmax[0],
                    self.negbintheta_spl_clamp_minmax[1]
                )


    def _check_args(self):
        '''
        Check args and raise appropriate error.
        :return:
        '''

        assert isinstance(self.coef_zinb_int_loglik, float)
        assert (self.coef_zinb_int_loglik >= 0.0)

        assert isinstance(self.coef_zinb_spl_loglik, float)
        assert (self.coef_zinb_spl_loglik >= 0.0)


        # the negbin_theta initvals, train flags, and minmax ===
        if not isinstance(self.initval_thetanegbin_int, float):
            assert(self.initval_thetanegbin_int == 'rand')

        if not isinstance(self.initval_thetanegbin_spl, float):
            assert(self.initval_thetanegbin_spl == 'rand')

        if not isinstance(self.negbintheta_int_clamp_minmax, list):
            assert (self.negbintheta_int_clamp_minmax is None)
            if self.flag_train_negbintheta_int:
                raise Exception(
                    "Since theta_negbin_int is set to trainable, min/max clip values must be provided for negbintheta_int."
                )
        else:
            assert (len(self.negbintheta_int_clamp_minmax) == 2)
            for u in self.negbintheta_int_clamp_minmax:
                assert (isinstance(u, float))
            if not self.flag_train_negbintheta_int:
                raise Exception(
                    "Since theta_negbin_int is not trainable, the min/max clip values must be set to None."
                )

        if not isinstance(self.negbintheta_spl_clamp_minmax, list):
            assert (self.negbintheta_spl_clamp_minmax is None)
            if self.flag_train_negbintheta_spl:
                raise Exception(
                    "Since theta_negbin_int is set to trainable, min/max clip values must be provided for negbintheta_int."
                )
        else:
            assert (len(self.negbintheta_spl_clamp_minmax) == 2)
            for u in self.negbintheta_spl_clamp_minmax:
                assert (isinstance(u, float))
            if not self.flag_train_negbintheta_spl:
                raise Exception(
                    "Since theta_negbin_int is not trainable, the min/max clip values must be set to None."
                )


        if not self.flag_use_int_u:
            assert (self.module_int_mu_u  is None)
            assert (self.module_int_cov_u is None)
        else:
            assert (isinstance(self.module_int_mu_u,  nn.Module))
            # if hasattr(self.module_int_mu_u, 'flag_endwithReLU'):
            assert (self.module_int_mu_u.flag_endwithReLU == False)
            assert (isinstance(self.module_int_cov_u, mlp.SimpleMLPandExp) or isinstance(self.module_int_cov_u, mlp.ConstMLP))

        if not self.flag_use_spl_u:
            assert (self.module_spl_mu_u  is None)
            assert (self.module_spl_cov_u is None)
        else:
            assert (isinstance(self.module_spl_mu_u,  nn.Module))
            assert (self.module_spl_mu_u.flag_endwithReLU == False)
            assert (isinstance(self.module_spl_cov_u, mlp.SimpleMLPandExp) or isinstance(self.module_spl_cov_u, mlp.ConstMLP))

        assert (
            self.dict_pname_to_scaleandunweighted.keys() == {
                'z', 'sin', 'sout', 'xbar_int', 'xbar_spl', 'x'
            }
        )
        for k in ['z', 'sin', 'sout', 'xbar_int', 'xbar_spl']:
            assert (isinstance(self.dict_pname_to_scaleandunweighted[k], list))
            assert (len(self.dict_pname_to_scaleandunweighted[k]) == 2)
            assert (self.dict_pname_to_scaleandunweighted[k][1] in [True, False])
        assert (
           isinstance(self.dict_pname_to_scaleandunweighted['x'], list)
        )
        assert (
            len(self.dict_pname_to_scaleandunweighted['x']) == 2
        )
        if None in self.dict_pname_to_scaleandunweighted['x']:
            assert (
                self.dict_pname_to_scaleandunweighted['x'] == [None, None]
            )
        else:
            assert (isinstance(self.dict_pname_to_scaleandunweighted['x'][0], float))
            assert (self.dict_pname_to_scaleandunweighted['x'][1] in [True, False])

        if not isinstance(self.initval_thetanegbin_int, float):
            assert (isinstance(self.initval_thetanegbin_int, str))
            assert(self.initval_thetanegbin_int == 'rand')

        if not isinstance(self.initval_thetanegbin_spl, float):
            assert (isinstance(self.initval_thetanegbin_spl, str))
            assert(self.initval_thetanegbin_spl == 'rand')

        # assert (
        #     isinstance(
        #         self.initval_thetanegbin_int, float
        #     )
        # )
        # assert (
        #     isinstance(
        #         self.initval_thetanegbin_spl, float
        #     )
        # )
        if isinstance(self.module_w_dec_int, mlp.SimpleMLP):
            if (not self.module_w_dec_int.flag_endwithReLU) and (not self.module_w_dec_int.flag_endswithSoftmax) and (not self.module_w_dec_int.flag_endswith_softplus):
                raise Exception(
                    "Set flag_endwithReLU to True, so the NegBin parameters are non-negative."
                )
        if isinstance(self.module_w_dec_spl, mlp.SimpleMLP):
            if (not self.module_w_dec_int.flag_endwithReLU) and (not self.module_w_dec_int.flag_endswithSoftmax) and (not self.module_w_dec_int.flag_endswith_softplus):
                raise Exception(
                    "Set flag_endwithReLU to True, so the NegBin parameters are non-negative."
                )


    @torch.no_grad()
    def sample(
        self,
        edge_index,
        t_num_steps:int,
        device,
        batch_size_feedforward,
        kwargs_dl_neighbourloader,
        ten_CT: torch.Tensor,
        ten_BatchEmb_in:torch.Tensor
    ):
        """
        :param edge_index:
        :param t_num_steps:
        :param device:
        :param batch_size_feedforward:
        :param kwargs_dl_neighbourloader:
        :param ten_CT:s
        :param ten_BatchEmb_in:
        :param np_size_factor:
        :return:
        """
        ten_u_int = (ten_CT + 0) if (self.flag_use_int_u) else None
        ten_u_spl = (ten_CT + 0) if (self.flag_use_spl_u) else None


        if pyg.utils.contains_self_loops(edge_index):
            raise Exception(
                "The provided graph contains seelf loops. Please ensure it doesn't have them and try again."
            )

        if not torch.all(
            torch.eq(
                torch.tensor(edge_index),
                pyg.utils.to_undirected(edge_index)
            )
        ):
            raise Exception(
                "The provided graph may contain seelf loops? If so, please ensure it doesn't have them and try again."
            )

        if self.flag_use_int_u:
            assert (ten_u_int is not None)
            assert (isinstance(ten_u_int, torch.Tensor))
        else:
            assert (ten_u_int is None)

        if self.flag_use_spl_u:
            assert (ten_u_spl is not None)
            assert (isinstance(ten_u_spl, torch.Tensor))
        else:
            assert (ten_u_spl is None)

        if not self.flag_use_spl_u:
            s_out = probutils.ExtenededNormal(
                loc=torch.zeros([self.num_cells, self.dict_varname_to_dim['s']]),
                scale=self.dict_pname_to_scaleandunweighted['sout'][0],
                flag_unweighted=self.dict_pname_to_scaleandunweighted['sout'][1]
            ).sample().to(device)  # [num_cell, dim_s]
        else:
            spl_cov_u = self.module_spl_cov_u(ten_u_spl)

            if isinstance(spl_cov_u, float):  # the case where the covariance is set to, e.g. 0.0 --> ExtendedNormal
                s_out = probutils.ExtenededNormal(
                    loc=self.module_spl_mu_u(ten_u_spl),
                    scale=np.sqrt(spl_cov_u),
                    flag_unweighted=self.dict_pname_to_scaleandunweighted['sout'][1]
                ).sample().to(device)  # [num_cell, dim_s]
            else:  # covariance is of the same shape as mu --> Normal
                assert (isinstance(spl_cov_u, torch.Tensor))
                s_out = probutils.Normal(
                    loc=self.module_spl_mu_u(ten_u_spl),
                    scale=spl_cov_u.sqrt()
                ).sample().to(device)  # [num_cell, dim_s]

        s_in = probutils.ExtenededNormal(
            loc=self.module_theta_aggr.evaluate_layered(
                x=s_out,
                edge_index=edge_index,
                kwargs_dl=kwargs_dl_neighbourloader
            ),
            scale=self.dict_pname_to_scaleandunweighted['sin'][0],
            flag_unweighted=self.dict_pname_to_scaleandunweighted['sin'][1]
        ).sample().to(device) # [num_cell, dim_s]


        if not self.flag_use_int_u:
            z = probutils.ExtenededNormal(
                loc=torch.zeros([self.num_cells, self.dict_varname_to_dim['z']]),
                scale=self.dict_pname_to_scaleandunweighted['z'][0],
                flag_unweighted=self.dict_pname_to_scaleandunweighted['z'][1]
            ).sample().to(device)  # [num_cell, dim_z]
        else:
            int_cov_u = self.module_int_cov_u(ten_u_int)

            if isinstance(int_cov_u, float):  # the case where int_cov_u is set to, e.g., 0.0 --> ExtendedNormal
                z = probutils.ExtenededNormal(
                    loc=self.module_int_mu_u(ten_u_int),
                    scale=np.sqrt(int_cov_u),
                    flag_unweighted=self.dict_pname_to_scaleandunweighted['z'][1]
                ).sample().to(device)  # [num_cell, dim_z]
            else:  # int_cov_u is like the iVAE paper --> Normal
                z = probutils.Normal(
                    loc=self.module_int_mu_u(ten_u_int),
                    scale=int_cov_u.sqrt()
                ).sample().to(device)  # [num_cell, dim_z]


        # TODO: is the below part needed?
        '''
        OLD: no conddist on the output of neural ODE
        output_neuralODE = probutils.ExtenededNormal(
            loc=utils.func_feed_x_to_neuralODEmodule(
                module_input=self.module_flow,
                x=torch.cat([z, s_in], 1),
                batch_size=batch_size_feedforward,
                t_span=torch.linspace(0, 1, t_num_steps).to(device)
            ),
            scale=torch.sqrt(torch.tensor(self.dict_sigma2s['sigma2_neuralODE'])),
            flag_unweighted=True
        ).sample().to(device)  # [num_cell, dim_z+dim_s]
        '''

        output_neuralODE = self.module_flow(
            t_in=torch.linspace(0, 1, t_num_steps).to(device),
            x_in=torch.cat(
                [z, s_in],
                1
            ),
            ten_BatchEmb_in=ten_BatchEmb_in
        )


        xbar_int = probutils.ExtenededNormal(
            loc=output_neuralODE[:, 0:self.dict_varname_to_dim['z']],
            scale=self.dict_pname_to_scaleandunweighted['xbar_int'][0],
            flag_unweighted=self.dict_pname_to_scaleandunweighted['xbar_int'][1]
        ).sample().to(device)  # [num_cells, dim_z]

        xbar_spl = probutils.ExtenededNormal(
            loc=output_neuralODE[:, self.dict_varname_to_dim['z']::],
            scale=self.dict_pname_to_scaleandunweighted['xbar_spl'][0],
            flag_unweighted=self.dict_pname_to_scaleandunweighted['xbar_spl'][1]
        ).sample().to(device)  # [num_cells, dim_s]


        # get the sotfmax mean values for Xint and Xspl ===
        x_int_softmax = utils.func_feed_x_to_module(
            module_input=self.module_w_dec_int,
            x=torch.cat(
                [ten_BatchEmb_in, xbar_int],
                1
            ),
            batch_size=batch_size_feedforward
        )
        x_spl_softmax = utils.func_feed_x_to_module(
            module_input=self.module_w_dec_spl,
            x=torch.cat(
                [ten_BatchEmb_in, xbar_spl],
                1
            ),
            batch_size=batch_size_feedforward
        )

        dict_toret = dict(
            ten_u_int=ten_u_int,
            ten_u_spl=ten_u_spl,
            s_out=s_out,
            s_in=s_in,
            z=z,
            xbar_int=xbar_int,
            xbar_spl=xbar_spl,
            x_int_softmax=x_int_softmax,
            x_spl_softmax=x_spl_softmax
        )
        return dict_toret

    def log_prob(self, dict_qsamples, batch, t_num_steps:int, np_size_factor:np.ndarray, coef_anneal):
        '''

        :param coef_anneal:
        :param dict_qsamples: samples from q.
        :param batch: the batch returned by pyg's neighborloader.
        :param t_num_steps: the number of time-steps to be used by the NeuralODE module.
        :param np_size_factor: size factor for each cell, a tensor of size [num_cells]
        :return:
        '''
        device = dict_qsamples['z'].device


        # s_out
        if not self.flag_use_spl_u:
            logp_s_out = probutils.ExtenededNormal(
                loc=torch.zeros([dict_qsamples['s_out'].size()[0], self.dict_varname_to_dim['s']]).to(device),
                scale=self.dict_pname_to_scaleandunweighted['sout'][0],
                flag_unweighted=self.dict_pname_to_scaleandunweighted['sout'][1]
            ).log_prob(dict_qsamples['s_out'])  # [num_cells, dim_s]
            spl_cov_u = None
        else:
            spl_cov_u = self.module_spl_cov_u(dict_qsamples['ten_u_spl'])

            if isinstance(spl_cov_u, float):  # the case where spl_cov_u is set to, e.g., 0.0 --> ExtendedNormal
                logp_s_out = probutils.ExtenededNormal(
                    loc=self.module_spl_mu_u(dict_qsamples['ten_u_spl']),
                    scale=np.sqrt(spl_cov_u),
                    flag_unweighted=self.dict_pname_to_scaleandunweighted['sout'][1]
                ).log_prob(dict_qsamples['s_out'])  # [num_cell, dim_s]
            else:  # spl_cov_u is like iVAE paper --> Normals
                assert (isinstance(spl_cov_u, torch.Tensor))
                logp_s_out = probutils.Normal(
                    loc=self.module_spl_mu_u(dict_qsamples['ten_u_spl']),
                    scale=spl_cov_u.sqrt()
                ).log_prob(dict_qsamples['s_out'])  # [num_cell, dim_s]


        # s_in
        logp_s_in = probutils.ExtenededNormal(
            loc=self.module_theta_aggr(
                x=dict_qsamples['s_out'],
                edge_index=batch.edge_index.to(device)
            )[:batch.batch_size],
            scale=self.dict_pname_to_scaleandunweighted['sin'][0],
            flag_unweighted=self.dict_pname_to_scaleandunweighted['sin'][1]
        ).log_prob(dict_qsamples['s_in'][:batch.batch_size])  # [b, dim_s] TODO: what if all instances are included ???

        # z
        if not self.flag_use_int_u:
            logp_z = probutils.ExtenededNormal(
                loc=torch.zeros([dict_qsamples['z'].size()[0], self.dict_varname_to_dim['z']]).to(device),
                scale=self.dict_pname_to_scaleandunweighted['z'][0],
                flag_unweighted=self.dict_pname_to_scaleandunweighted['z'][1]
            ).log_prob(dict_qsamples['z'])  # [num_cells, dim_z]
            int_cov_u = None
        else:

            int_cov_u = self.module_int_cov_u(dict_qsamples['ten_u_int'])

            if isinstance(int_cov_u, float):  # the case where int_cov_u is, e.g., 0.0 --> ExtendedNormal
                logp_z = probutils.ExtenededNormal(
                    loc=self.module_int_mu_u(dict_qsamples['ten_u_int']),
                    scale=np.sqrt(int_cov_u),
                    flag_unweighted=self.dict_pname_to_scaleandunweighted['z'][1]
                ).log_prob(dict_qsamples['z'])  # [num_cells, dim_z]
            else:  # int_cov_u is like the iVAE paper --> Normal
                assert (isinstance(int_cov_u, torch.Tensor))
                logp_z = probutils.Normal(
                    loc=self.module_int_mu_u(dict_qsamples['ten_u_int']),
                    scale=int_cov_u.sqrt()
                ).log_prob(dict_qsamples['z'])  # [num_cells, dim_z]

        # annealing
        if coef_anneal is not None:
            logp_s_out = logp_s_out * coef_anneal
            logp_z = logp_z * coef_anneal



        # xbar_int, xbar_spl
        rng_batchemb = [
            batch.MINTFLOWMETAINF['dim_u_int'] + batch.MINTFLOWMETAINF['dim_u_spl'] + batch.MINTFLOWMETAINF['dim_CT'] + batch.MINTFLOWMETAINF['dim_NCC'],
            batch.MINTFLOWMETAINF['dim_u_int'] + batch.MINTFLOWMETAINF['dim_u_spl'] + batch.MINTFLOWMETAINF['dim_CT'] + batch.MINTFLOWMETAINF['dim_NCC'] + batch.MINTFLOWMETAINF['dim_BatchEmb']
        ]
        '''
        output_neuralODE = self.module_flow(
            torch.cat(
                [
                    batch.y[:, rng_batchemb[0]:rng_batchemb[1]][:batch.batch_size].float().to(device),
                    dict_qsamples['z'][:batch.batch_size],
                    dict_qsamples['s_in'][:batch.batch_size]
                ],
                1
            ),
            torch.linspace(0, 1, t_num_steps).to(device)
        )[1][-1, :, :]  # [b, dim_z+dim_s]
        '''
        output_neuralODE = self.module_flow(
            t_in=torch.linspace(0, 1, t_num_steps).to(device),
            x_in=torch.cat(
                [
                    dict_qsamples['z'][:batch.batch_size],
                    dict_qsamples['s_in'][:batch.batch_size]
                ],
                1
            ),
            ten_BatchEmb_in=batch.y[:, rng_batchemb[0]:rng_batchemb[1]][:batch.batch_size].float().to(device)
        )

        logp_xbarint = probutils.ExtenededNormal(
            loc=output_neuralODE[:, 0:self.dict_varname_to_dim['z']],
            scale=self.dict_pname_to_scaleandunweighted['xbar_int'][0],
            flag_unweighted=self.dict_pname_to_scaleandunweighted['xbar_int'][1]
        ).log_prob(
            dict_qsamples['xbar_int'][:batch.batch_size]
        )  # [b, dim_z]

        logp_xbarspl = probutils.ExtenededNormal(
            loc=output_neuralODE[:, self.dict_varname_to_dim['z']::],
            scale=self.dict_pname_to_scaleandunweighted['xbar_spl'][0],
            flag_unweighted=self.dict_pname_to_scaleandunweighted['xbar_spl'][1]
        ).log_prob(
            dict_qsamples['xbar_spl'][:batch.batch_size]
        )  # [b, dim_s]


        # FLAG_MODE_DEC_SIZEFACTOR_XSUM ====
        FLAG_MODE_DEC_SIZEFACTOR_XSUM = True

        # x_int
        netout_w_dec_int = self.module_w_dec_int(
            torch.cat(
                [
                    batch.y[:, rng_batchemb[0]:rng_batchemb[1]][:batch.batch_size].float().to(device),
                    dict_qsamples['xbar_int'][:batch.batch_size]
                ],
                1
            )
        )

        if FLAG_MODE_DEC_SIZEFACTOR_XSUM:
            with torch.no_grad():
                sizefactor_int = dict_qsamples['x_int'][:batch.batch_size].sum(1).unsqueeze(-1)  # [b, num_genes]
        else:
            sizefactor_int = torch.tensor(np_size_factor[batch.input_id], device=device, requires_grad=False).unsqueeze(1)

        logp_x_int = self.coef_zinb_int_loglik * ZeroInflatedNegativeBinomial(
            **{**{'mu': netout_w_dec_int * sizefactor_int.detach(),
                  'theta': torch.exp(self.theta_negbin_int)},
                  **self.kwargs_negbin_int}
        ).log_prob(dict_qsamples['x_int'][:batch.batch_size])  # [b, num_genes]


        # x_spl
        netout_w_dec_spl = self.module_w_dec_spl(
            torch.cat(
                [
                    batch.y[:, rng_batchemb[0]:rng_batchemb[1]][:batch.batch_size].float().to(device),
                    dict_qsamples['xbar_spl'][:batch.batch_size]
                ],
                1
            )
        )

        if FLAG_MODE_DEC_SIZEFACTOR_XSUM:
            with torch.no_grad():
                sizefactor_spl = dict_qsamples['x_spl'][:batch.batch_size].sum(1).unsqueeze(-1)  # [b, num_genes]
        else:
            sizefactor_spl = torch.tensor(np_size_factor[batch.input_id], device=device, requires_grad=False).unsqueeze(1)

        logp_x_spl = self.coef_zinb_spl_loglik * ZeroInflatedNegativeBinomial(
            **{**{'mu': netout_w_dec_spl * sizefactor_spl.detach(),
                  'theta': torch.exp(self.theta_negbin_spl)},
               **self.kwargs_negbin_spl}
        ).log_prob(dict_qsamples['x_spl'][:batch.batch_size])  # [b, num_genes]

        # x
        if None in self.dict_pname_to_scaleandunweighted['x']:
            # no logp(x | x_int, x_spl) term
            assert ( self.dict_pname_to_scaleandunweighted['x'] == [None, None])
            logp_x = torch.tensor([[1.0]], device=logp_x_spl.device, requires_grad=False)
        else:
            logp_x = probutils.ExtenededNormal(
                loc=dict_qsamples['x_int'][:batch.batch_size]+dict_qsamples['x_spl'][:batch.batch_size],
                scale=self.dict_pname_to_scaleandunweighted['x'][0],
                flag_unweighted=self.dict_pname_to_scaleandunweighted['x'][1]
            ).log_prob(batch.x.to_dense()[:batch.batch_size].to(device))  # [b, num_genes]

        dict_logp = dict(
            logp_s_out=logp_s_out,
            logp_z=logp_z,
            logp_s_in=logp_s_in,
            logp_xbarint=logp_xbarint,
            logp_xbarspl=logp_xbarspl,
            logp_x_int=logp_x_int,
            logp_x_spl=logp_x_spl,
            logp_x=logp_x
        )
        dict_otherinf = dict(
            int_cov_u=int_cov_u,
            spl_cov_u=spl_cov_u,
            coef_anneal=coef_anneal
        )
        return dict_logp, dict_otherinf
























