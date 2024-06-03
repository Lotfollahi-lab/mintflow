
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from .generativemodel import InFlowGenerativeModel
from .modules.impanddisentgl import  ImputerAndDisentangler
from .modules.cond4flow import Cond4FlowVarphi0
from . import probutils
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter


class InFlowVarDist(nn.Module):
    '''
    Variational distribution for inflow.
    '''
    def __init__(
            self,
            module_genmodel:InFlowGenerativeModel,
            kwargs_impanddisentgl:dict,
            module_varphi_enc_int:nn.Module,
            module_varphi_enc_spl:nn.Module,
            kwargs_cond4flowvarphi0:dict,
            dict_qname_to_scaleandunweighted:dict
    ):
        '''

        :param module_genmodel: the generative model
            TODO: double-check: Important note: in synthetic setting two instances of the generative model are used
                - the one used to generate the observations (not passed here).
                - the one passed here, which has the same architecture to the prev but is has different params.
        :param kwargs_impanddisentgl:
            To be used to instantiate `ImputerAndDisentangler`.
        :module_varphi_enc_int, module_varphi_enc_int: the encoder modules to map gene expression vectors to a
            lower-dimensional embedding space, denoted by \varphi_enc in the paper.
            These modules can be trained separated with, e.g., PCA.
            A single module can be passed as both these arguments.
        :param kwargs_cond4flowvarphi0: to be used to instantiate `Cond4FlowVarphi0`.
        :param dict_qname_to_scaleandunweighted: a dictionary to decide
            1. sigma-s #TODO: complete the doc
            2. unweighted flags: #TODO: complete the doc
            In summary, scale=0 and flag_unweight=False means it's deterministic.
            #TODO: now sigmas are set manually . Add learnable sigmas.
            - this dictionary has the follwoing keys
                - impanddisentgl_int: with keys scale, flag_unweighted
                - impanddisentgl_spl: with keys scale, flag_unweighted
                - varphi_enc_int: with keys scale, flag_unweighted
                - varphi_enc_spl: with keys scale, flag_unweighted
                - z: with keys scale, flag_unweighted
                - sin: with keys scale, flag_unweighted
                - sout: with keys scale, flag_unweighted

        '''
        super(InFlowVarDist, self).__init__()
        # grab args ===
        self.module_genmodel = module_genmodel
        self.kwargs_impanddisentgl = kwargs_impanddisentgl
        self.module_varphi_enc_int = module_varphi_enc_int
        self.module_varphi_enc_spl = module_varphi_enc_spl
        self.kwargs_cond4flowvarphi0 = kwargs_cond4flowvarphi0
        self.dict_qname_to_scaleandunweighted = dict_qname_to_scaleandunweighted
        self._check_args()

        # make internals
        self.module_impanddisentgl = ImputerAndDisentangler(**kwargs_impanddisentgl)
        self.module_cond4flowvarphi0 = Cond4FlowVarphi0(**kwargs_cond4flowvarphi0)

    def rsample(self, batch, prob_maskknowngenes:float, ten_xy_absolute:torch.Tensor):
        # step 1, rsample from imputer and disentangler
        params_q_impanddisentgl = self.module_impanddisentgl(
            batch=batch,
            prob_maskknowngenes=prob_maskknowngenes,
            ten_xy_absolute=ten_xy_absolute
        )  # so it's not repeated in compgraph.
        x_int = probutils.ExtenededNormal(
            loc=params_q_impanddisentgl['muxint'],
            scale=self.dict_qname_to_scaleandunweighted['impanddisentgl_int']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['impanddisentgl_int']['flag_unweighted']
        ).rsample()  # [N, num_genes]
        x_spl = probutils.ExtenededNormal(
            loc=params_q_impanddisentgl['muxspl'],
            scale=self.dict_qname_to_scaleandunweighted['impanddisentgl_spl']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['impanddisentgl_spl']['flag_unweighted']
        ).rsample()  # [N, num_genes]

        # step 2, rsample from encoders to the low-dim embedding space.
        param_q_xbarint = self.module_varphi_enc_int(x_int)  # [N, dim_latent]
        param_q_xbarspl = self.module_varphi_enc_spl(x_spl)  # [N, dim_latent]
        xbar_int = probutils.ExtenededNormal(
            loc=param_q_xbarint,
            scale=self.dict_qname_to_scaleandunweighted['varphi_enc_int']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['varphi_enc_int']['flag_unweighted']
        ).rsample()  # [N, dim_latent]
        xbar_spl = probutils.ExtenededNormal(
            loc=param_q_xbarspl,
            scale=self.dict_qname_to_scaleandunweighted['varphi_enc_spl']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['varphi_enc_spl']['flag_unweighted']
        ).rsample()  # [N, dim_latent]

        # step 3, rsample from z, s_in, and s_out
        param_q_cond4flow = self.module_cond4flowvarphi0(
            ten_xbar_int=xbar_int,
            batch=batch,
            ten_xbar_spl=xbar_spl,
            ten_xy_absolute=ten_xy_absolute
        )
        z = probutils.ExtenededNormal(
            loc=param_q_cond4flow['mu_z'],
            scale=self.dict_qname_to_scaleandunweighted['z']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['z']['flag_unweighted']
        ).rsample()  # [N, dim_z]
        s_in = probutils.ExtenededNormal(
            loc=param_q_cond4flow['mu_sin'],
            scale=self.dict_qname_to_scaleandunweighted['sin']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['sin']['flag_unweighted']
        ).rsample()  # [N, dim_s]
        s_out = probutils.ExtenededNormal(
            loc=param_q_cond4flow['mu_sout'],
            scale=self.dict_qname_to_scaleandunweighted['sout']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['sout']['flag_unweighted']
        ).rsample()  # [N, dim_s]

        # ret
        return dict(
            params_q_impanddisentgl=params_q_impanddisentgl,
            param_q_xbarint=param_q_xbarint,
            param_q_xbarspl=param_q_xbarspl,
            param_q_cond4flow=param_q_cond4flow,
            x_int=x_int,
            x_spl=x_spl,
            xbar_int=xbar_int,
            xbar_spl=xbar_spl,
            z=z,
            s_in=s_in,
            s_out=s_out,
            loss_imputex=params_q_impanddisentgl['loss_imputex'],
            ten_out_imputer=params_q_impanddisentgl['ten_out_imputer']
        )

    def log_prob(self, dict_retvalrsample):
        # xint
        logq_xint = probutils.ExtenededNormal(
            loc=dict_retvalrsample['params_q_impanddisentgl']['muxint'],
            scale=self.dict_qname_to_scaleandunweighted['impanddisentgl_int']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['impanddisentgl_int']['flag_unweighted']
        ).log_prob(dict_retvalrsample['x_int'])  # [N, num_genes]

        # xspl
        logq_xspl = probutils.ExtenededNormal(
            loc=dict_retvalrsample['params_q_impanddisentgl']['muxspl'],
            scale=self.dict_qname_to_scaleandunweighted['impanddisentgl_spl']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['impanddisentgl_spl']['flag_unweighted']
        ).log_prob(dict_retvalrsample['x_spl'])  # [N, num_genes]

        # xbarint
        logq_xbarint = probutils.ExtenededNormal(
            loc=dict_retvalrsample['param_q_xbarint'],
            scale=self.dict_qname_to_scaleandunweighted['varphi_enc_int']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['varphi_enc_int']['flag_unweighted']
        ).log_prob(dict_retvalrsample['xbar_int'])  # [N, dim_latent]

        # xbarspl
        logq_xbarspl = probutils.ExtenededNormal(
            loc=dict_retvalrsample['param_q_xbarspl'],
            scale=self.dict_qname_to_scaleandunweighted['varphi_enc_spl']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['varphi_enc_spl']['flag_unweighted']
        ).log_prob(dict_retvalrsample['xbar_spl'])  # [N, dim_latent]

        # z, s_in, s_out
        logq_z = probutils.ExtenededNormal(
            loc=dict_retvalrsample['param_q_cond4flow']['mu_z'],
            scale=self.dict_qname_to_scaleandunweighted['z']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['z']['flag_unweighted']
        ).log_prob(dict_retvalrsample['z'])  # [N, dim_z]
        logq_s_in = probutils.ExtenededNormal(
            loc=dict_retvalrsample['param_q_cond4flow']['mu_sin'],
            scale=self.dict_qname_to_scaleandunweighted['sin']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['sin']['flag_unweighted']
        ).log_prob(dict_retvalrsample['s_in'])  # [N, dim_s]
        logq_s_out = probutils.ExtenededNormal(
            loc=dict_retvalrsample['param_q_cond4flow']['mu_sout'],
            scale=self.dict_qname_to_scaleandunweighted['sout']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['sout']['flag_unweighted']
        ).log_prob(dict_retvalrsample['s_out'])  # [N, dim_s]

        # ret
        return dict(
            logq_xint=logq_xint,
            logq_xspl=logq_xspl,
            logq_xbarint=logq_xbarint,
            logq_xbarspl=logq_xbarspl,
            logq_z=logq_z,
            logq_s_in=logq_s_in,
            logq_s_out=logq_s_out
        )


    def training_epoch(
            self,
            dl:NeighborLoader,
            prob_maskknowngenes:float,
            t_num_steps:int,
            ten_xy_absolute:torch.Tensor,
            optim_training:torch.optim.Optimizer,
            tensorboard_stepsize_save:int,
            log_dir_tensorboard=None,
            sum_writer_input:SummaryWriter|None=None,
            itrcount_tensorboard_input:int=0
    ):
        '''
        One epoch of the training.
        :param dl: pyg's neighborloader.
        :param prob_maskknowngenes: probability of masking some known gene expressions (per-cell)
            to define the self-supervised loss.
        :param t_num_steps: the number of time-steps to be used by the NeuralODE module.
        :param ten_xy_absolute: absoliute xy positions of all cells.
        :param optim_training: the optimizer.
        :return:
        '''

        if log_dir_tensorboard is not None:
            if sum_writer_input is None:
                sum_writer = SummaryWriter(log_dir_tensorboard)
                itrcount_tensorboard = 0
            else:
                sum_writer = sum_writer_input
                itrcount_tensorboard = itrcount_tensorboard_input + 0
        else:
            sum_writer = None
            itrcount_tensorboard = 0


        for batch in tqdm(dl):
            optim_training.zero_grad()
            flag_tensorboardsave = (itrcount_tensorboard%tensorboard_stepsize_save == 0)\
                                    and (log_dir_tensorboard is not None)

            dict_q_sample = self.rsample(
                batch=batch,
                prob_maskknowngenes=prob_maskknowngenes,
                ten_xy_absolute=ten_xy_absolute
            )
            dict_logq = self.log_prob(dict_q_sample)
            dict_logp = self.module_genmodel.log_prob(
                dict_qsamples=dict_q_sample,
                batch=batch,
                t_num_steps=t_num_steps
            )

            # make the loss
            loss = 0.0
            for k in dict_logp.keys():
                loss = loss - dict_logp[k].sum(1).mean()
                if flag_tensorboardsave:
                    with torch.no_grad():
                        sum_writer.add_scalar(
                            "Loss/logprob_P/{}".format(k),
                            torch.mean(- dict_logp[k].sum(1).mean()),
                            itrcount_tensorboard
                        )

            for k in dict_logq.keys():
                loss = loss + dict_logq[k].sum(1).mean()
                if flag_tensorboardsave:
                    with torch.no_grad():
                        sum_writer.add_scalar(
                            "Loss/logprob_Q/{}".format(k),
                            torch.mean(+ dict_logq[k].sum(1).mean()),
                            itrcount_tensorboard
                        )

            # add the imputation loss
            if dict_q_sample['loss_imputex'] is not None:
                loss = loss + dict_q_sample['loss_imputex'].mean()

            if flag_tensorboardsave:
                with torch.no_grad():
                    if dict_q_sample['loss_imputex'] is not None:
                        sum_writer.add_scalar(
                            "Loss/loss_imputex",
                            dict_q_sample['loss_imputex'].mean(),
                            itrcount_tensorboard
                        )
                    else:
                        sum_writer.add_scalar(
                            "Loss/loss_imputex",
                            torch.nan,
                            itrcount_tensorboard
                        )


            if flag_tensorboardsave:
                sum_writer.flush()

            # update params
            loss.backward()
            optim_training.step()
            itrcount_tensorboard += 1

        return sum_writer, itrcount_tensorboard



    @torch.no_grad()
    def eval_on_pygneighloader_dense(self, dl:NeighborLoader, ten_xy_absolute:torch.Tensor):
        '''
        Evaluates the model on a pyg.NeighborLoader.
        All results are obtained in dense arrays and returned.
        :param dl:
        :param ten_xy_absolute:
        :return:
        '''
        self.eval()
        dict_var_to_dict_nglobal_to_value = {
            'output_imputer':{},
            'muxint':{},
            'muxspl':{},
            'muxbar_int':{},
            'muxbar_spl':{},
            'mu_sin':{},
            'mu_sout':{},
            'mu_z':{}
        }  # TODO: add other variables.
        for batch in tqdm(dl):
            curr_dict_qsample = self.rsample(
                batch=batch,
                prob_maskknowngenes=0.0,
                ten_xy_absolute=ten_xy_absolute
            )
            np_out_imputer = curr_dict_qsample['ten_out_imputer'].detach().cpu().numpy()
            np_muxint = curr_dict_qsample['params_q_impanddisentgl']['muxint'].detach().cpu().numpy()
            np_muxspl = curr_dict_qsample['params_q_impanddisentgl']['muxspl'].detach().cpu().numpy()
            np_muxbar_int = curr_dict_qsample['param_q_xbarint'].detach().cpu().numpy()
            np_muxbar_spl = curr_dict_qsample['param_q_xbarspl'].detach().cpu().numpy()
            np_mu_sin = curr_dict_qsample['param_q_cond4flow']['mu_sin'].detach().cpu().numpy()
            np_mu_sout = curr_dict_qsample['param_q_cond4flow']['mu_sout'].detach().cpu().numpy()
            np_mu_z = curr_dict_qsample['param_q_cond4flow']['mu_z'].detach().cpu().numpy()
            for n_local, n_global in enumerate(batch.input_id.tolist()):
                dict_var_to_dict_nglobal_to_value['output_imputer'][n_global] = np_out_imputer[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxint'][n_global] = np_muxint[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxspl'][n_global] = np_muxspl[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxbar_int'][n_global] = np_muxbar_int[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxbar_spl'][n_global] = np_muxbar_spl[n_local, :]
                dict_var_to_dict_nglobal_to_value['mu_sin'][n_global] = np_mu_sin[n_local, :]
                dict_var_to_dict_nglobal_to_value['mu_sout'][n_global] = np_mu_sout[n_local, :]
                dict_var_to_dict_nglobal_to_value['mu_z'][n_global] = np_mu_z[n_local, :]

        self.train()

        # create dict_varname_to_output
        dict_varname_to_output = {}
        for k in dict_var_to_dict_nglobal_to_value.keys():
            assert(
                set(dict_var_to_dict_nglobal_to_value[k].keys()) == set(range(ten_xy_absolute.size()[0]))
            )
            dict_varname_to_output[k] = np.stack(
                [dict_var_to_dict_nglobal_to_value[k][n] for n in range(ten_xy_absolute.size()[0])],
                0
            )

        return dict_varname_to_output









    def _check_args(self):
        # check dict_qname_to_scaleandunweighted
        assert(
            set(self.dict_qname_to_scaleandunweighted.keys()) ==
            {'impanddisentgl_int', 'impanddisentgl_spl', 'varphi_enc_int', 'varphi_enc_spl', 'z', 'sin', 'sout'}
        )
        for k in self.dict_qname_to_scaleandunweighted.keys():
            assert(
                isinstance(self.dict_qname_to_scaleandunweighted[k], dict)
            )
            assert(
                set(self.dict_qname_to_scaleandunweighted[k].keys()) ==
                {'scale', 'flag_unweighted'}
            )









