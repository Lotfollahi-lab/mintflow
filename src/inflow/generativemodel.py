

import torch
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


class DummyModule(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(DummyModule, self).__init__()
    def forward(self, x):
        return x

class InFlowGenerativeModel(nn.Module):
    def __init__(
            self,
            num_cells,
            dict_varname_to_dim,
            dict_sigma2s,
            module_z_sampler: zs_samplers.ZSSampler,
            module_s_sampler: zs_samplers.ZSSampler,
            type_theta_aggr, kwargs_theta_aggr,
            type_moduleflow, kwargs_moduleflow,
            type_w_dec, kwargs_w_dec,
            kwargs_negbin_int, kwargs_negbin_spl
    ):
        '''

        :param dict_varname_to_dim: a dict with keys
            - s
            - z
            - x
            - TODO:maybe more
        :param dict_sigma2s containning
            - sigma2_aggr
            - sigma2_neuralODE
            - sigma2_decoder
            - sigma2_sum
        :param TODO:complete
        '''
        super(InFlowGenerativeModel, self).__init__()
        #grab args ===
        self.num_cells = num_cells
        self.dict_varname_to_dim = dict_varname_to_dim
        self.dict_sigma2s = dict_sigma2s
        self.module_z_sampler = module_z_sampler
        self.module_s_sampler = module_s_sampler
        self.kwargs_negbin_int, self.kwargs_negbin_spl = kwargs_negbin_int, kwargs_negbin_spl
        #check args ===
        assert(
            self.dict_sigma2s.keys() == {
                'sigma2_aggr', 'sigma2_neuralODE', 'sigma2_decoder', 'sigma2_sum'
            }
        )
        #make internals ===
        self.module_theta_aggr = type_theta_aggr(
            **{**{
                'dim_input':self.dict_varname_to_dim['s'],
                'dim_output':self.dict_varname_to_dim['s']},
                **kwargs_theta_aggr
            }
        ) #TODO: add note about requiring the `dim_input` and `dim_output` arguments.
        self.module_flow = NeuralODE(
            torch_wrapper(
                type_moduleflow(
                    **{**{
                        'dim_input': self.dict_varname_to_dim['s'] + self.dict_varname_to_dim['z'],
                        'dim_output': self.dict_varname_to_dim['s'] + self.dict_varname_to_dim['z']},
                       **kwargs_moduleflow
                    }
                )
            ),
            solver="dopri5",
            sensitivity="adjoint",
            return_t_eval=True
        )
        self.module_w_dec_int = type_w_dec(
            **{**{
                'dim_input': self.dict_varname_to_dim['z'],
                'dim_output': self.dict_varname_to_dim['x']},
               **kwargs_w_dec
               }
        )
        self.module_w_dec_spl = type_w_dec(
            **{**{
                'dim_input':self.dict_varname_to_dim['s'],
                'dim_output':self.dict_varname_to_dim['x']},
               **kwargs_w_dec
            }
        )

        #make the theta parameters of NegBin intrinsic and NegBin spatial
        self.theta_negbin_int = torch.nn.Parameter(
            0.5 * torch.ones(self.dict_varname_to_dim['x']).unsqueeze(0),
            requires_grad=False
        ) #TODO: tune #TODO: maybe change it to trianable?
        self.theta_negbin_spl = torch.nn.Parameter(
            0.5 * torch.ones(self.dict_varname_to_dim['x']).unsqueeze(0),
            requires_grad=False
        ) #TODO: tune #TODO: maybe change it to trianable?

        self._check_args()

    def _check_args(self):
        '''
        Check args and raise appropriate error.
        :return:
        '''
        pass
        if isinstance(self.module_w_dec_int, mlp.SimpleMLP):
            if not self.module_w_dec_int.flag_endwithReLU:
                raise Exception(
                    "Set flag_endwithReLU to True, so the NegBin parameters are non-negative."
                )
        if isinstance(self.module_w_dec_spl, mlp.SimpleMLP):
            if not self.module_w_dec_spl.flag_endwithReLU:
                raise Exception(
                    "Set flag_endwithReLU to True, so the NegBin parameters are non-negative."
                )


    @torch.no_grad()
    def sample(self, edge_index, t_num_steps:int, device, batch_size_feedforward, kwargs_dl_neighbourloader):
        '''
        Generates a single sample from all variables.
        :param edge_index: the edges of the neighbourhood graph, must not contain self-loops.
        :param t_num_steps: the number of time steps between 0 and 1 for the neural ODE.
        :param batch_size_feedforward: when `num_cells` is huge different representations are processed
            batch by batch, where each batch is determined by this argument.
        :return:
        '''
        assert(
            not pyg.utils.contains_self_loops(edge_index)
        )
        assert (
            torch.all(
                torch.eq(
                    torch.tensor(edge_index),
                    pyg.utils.to_undirected(edge_index)
                )
            )
        )
        s_out = self.module_s_sampler.gen_zs(
            N=self.num_cells,
            D=self.dict_varname_to_dim['s']
        ).to(device)  # [num_cell, dim_s]
        '''
        s_out = Normal(
            loc=torch.zeros([self.num_cells, self.dict_varname_to_dim['s']]),
            scale=torch.tensor([1.0])
        ).sample().to(device) # [num_cell, dim_s]
        '''
        s_in = probutils.ExtenededNormal(
            loc=self.module_theta_aggr.evaluate_layered(
                x=s_out,
                edge_index=edge_index,
                kwargs_dl=kwargs_dl_neighbourloader
            ),
            scale=torch.sqrt(torch.tensor(self.dict_sigma2s['sigma2_aggr'])),
            flag_unweighted=True
        ).sample().to(device) # [num_cell, dim_s]
        # #TODO: add description of `evaluate_layered` with x and edge_index signatures.
        #TODO: assert the evalu_layered function and args.

        z = self.module_z_sampler.gen_zs(
            N=self.num_cells,
            D=self.dict_varname_to_dim['z']
        ).to(device)  # [num_cell, dim_z]

        '''
        z = Normal(
            loc=torch.zeros([self.num_cells, self.dict_varname_to_dim['z']]),
            scale=torch.tensor([1.0])
        ).sample().to(device)  # [num_cell, dim_z]
        '''


        '''
        recall the output from neuralODE module is as follows
        - output[0]: is the t_range.
        - output[1]: is of shape [len(t_range), N, D].
        '''
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
        xbar_int = probutils.ExtenededNormal(
            loc=output_neuralODE[:, 0:self.dict_varname_to_dim['z']],
            scale=self.dict_sigma2s['sigma2_decoder'],
            flag_unweighted=True
        ).sample().to(device)  # [num_cells, dim_z]
        xbar_spl = probutils.ExtenededNormal(
            loc=output_neuralODE[:, self.dict_varname_to_dim['z']::],
            scale=self.dict_sigma2s['sigma2_decoder'],
            flag_unweighted=True
        ).sample().to(device)  # [num_cells, dim_s]



        #feed to NegativeBinomial distributions ===
        x_int = ZeroInflatedNegativeBinomial(
            **{**{'mu': utils.func_feed_x_to_module(
                    module_input=self.module_w_dec_int,
                    x=xbar_int,
                    batch_size=batch_size_feedforward),
                  'theta':self.theta_negbin_int},
                **self.kwargs_negbin_int}
        ).sample() #[num_cells, num_genes] #TODO: make the theta parameter learnable.
        x_spl = ZeroInflatedNegativeBinomial(
            **{**{'mu': utils.func_feed_x_to_module(
                    module_input=self.module_w_dec_spl,
                    x=xbar_spl,
                    batch_size=batch_size_feedforward),
                  'theta':self.theta_negbin_spl},
                **self.kwargs_negbin_spl}
        ).sample()  # [num_cells, num_genes] #TODO: make the theta parameter learnable.

        #generate x ===
        if self.dict_sigma2s['sigma2_sum'] != 0.0:
            raise Exception(
                "Please set dict_sigma2s['sigma2_sum'] to 0"+
                "otherwise the final generated x will have continuous values unless x_int and x_spl are continuous."
            )

        x = probutils.ExtenededNormal(
            loc=x_int+x_spl,
            scale=torch.sqrt(torch.tensor(self.dict_sigma2s['sigma2_sum'])),
            flag_unweighted=True
        ).sample()  # [num_cells, num_genes]

        dict_toret = dict(
            s_out=s_out,
            s_in=s_in,
            z=z,
            xbar_int=xbar_int,
            xbar_spl=xbar_spl,
            x_int=x_int,
            x_spl=x_spl,
            x=x
        )
        return dict_toret

    def log_prob(self, dict_qsamples, batch, t_num_steps:int):
        '''

        :param dict_qsamples: samples from q.
        :param batch: the batch returned by pyg's neighborloader.
        :param t_num_steps: the number of time-steps to be used by the NeuralODE module.
        :return:
        '''
        device = dict_qsamples['z'].device

        # s_out
        logp_s_out = Normal(
            loc=torch.zeros([dict_qsamples['s_out'].size()[0], self.dict_varname_to_dim['s']]).to(device),
            scale=torch.tensor([1.0]).to(device)
        ).log_prob(dict_qsamples['s_out'])  # [num_cells, dim_s]

        # s_in
        logp_s_in = probutils.ExtenededNormal(
            loc=self.module_theta_aggr(
                x=dict_qsamples['s_out'],
                edge_index=batch.edge_index.to(device)
            )[:batch.batch_size],
            scale=torch.sqrt(torch.tensor(self.dict_sigma2s['sigma2_aggr'])).to(device),
            flag_unweighted=True
        ).log_prob(dict_qsamples['s_in'][:batch.batch_size])  # [b, dim_s] TODO: what if all instances are included ???

        # z
        logp_z = Normal(
            loc=torch.zeros([dict_qsamples['z'].size()[0], self.dict_varname_to_dim['z']]).to(device),
            scale=torch.tensor([1.0]).to(device)
        ).log_prob(dict_qsamples['z'])  # [num_cells, dim_z]

        # xbar_int, xbar_spl
        output_neuralODE = self.module_flow(
            torch.cat(
                [dict_qsamples['z'][:batch.batch_size], dict_qsamples['s_in'][:batch.batch_size]],
                1
            ),
            torch.linspace(0, 1, t_num_steps).to(device)
        )[1][-1, :, :]  # [b, dim_z+dim_s]

        logp_xbarint = probutils.ExtenededNormal(
            loc=output_neuralODE[:, 0:self.dict_varname_to_dim['z']],
            scale=torch.tensor([self.dict_sigma2s['sigma2_decoder']]).to(device),
            flag_unweighted=True
        ).log_prob(
            dict_qsamples['xbar_int'][:batch.batch_size]
        )  # [b, dim_z]

        logp_xbarspl = probutils.ExtenededNormal(
            loc=output_neuralODE[:, self.dict_varname_to_dim['z']::],
            scale=torch.tensor(self.dict_sigma2s['sigma2_decoder']).to(device),
            flag_unweighted=True
        ).log_prob(
            dict_qsamples['xbar_spl'][:batch.batch_size]
        )  # [b, dim_s]



        # x_int
        logp_x_int = ZeroInflatedNegativeBinomial(
            **{**{'mu': self.module_w_dec_int(dict_qsamples['xbar_int'][:batch.batch_size]),
                  'theta': self.theta_negbin_int},
                  **self.kwargs_negbin_int}
        ).log_prob(dict_qsamples['x_int'][:batch.batch_size])  # [b, num_genes]

        # x_spl
        logp_x_spl = ZeroInflatedNegativeBinomial(
            **{**{'mu': self.module_w_dec_spl(dict_qsamples['xbar_spl'][:batch.batch_size]),
                  'theta': self.theta_negbin_spl},
               **self.kwargs_negbin_spl}
        ).log_prob(dict_qsamples['x_spl'][:batch.batch_size])  # [b, num_genes]

        # x
        logp_x = probutils.ExtenededNormal(
            loc=dict_qsamples['x_int'][:batch.batch_size]+dict_qsamples['x_spl'][:batch.batch_size],
            scale=torch.sqrt(torch.tensor(self.dict_sigma2s['sigma2_sum'])).to(device),
            flag_unweighted=True
        ).log_prob(batch.x.to_dense()[:batch.batch_size].to(device))  # [b, num_genes]

        return dict(
            logp_s_out=logp_s_out,
            logp_z=logp_z,
            logp_s_in=logp_s_in,
            logp_xbarint=logp_xbarint,
            logp_xbarspl=logp_xbarspl,
            logp_x_int=logp_x_int,
            logp_x_spl=logp_x_spl,
            logp_x=logp_x
        )
























