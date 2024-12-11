import random
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import NeighborLoader
from .generativemodel import InFlowGenerativeModel
from .modules.impanddisentgl import ImputerAndDisentangler
from .modules.disentonly import Disentangler
from .modules.cond4flow import Cond4FlowVarphi0
from .modules.cond4flow_simple3mpls import Cond4FlowVarphi0SimpleMLPs
from .modules.disentonly_twosep import DisentanglerTwoSep
from .modules.gnn_disentangler import GNNDisentangler
from torch.distributions.normal import Normal
from . import probutils
from . import utils_imputer
from . predadjmat import ListAdjMatPredLoss
from . import utils_flowmatching
from . import kl_annealing
from .modules import predictorperCT
import predadjmat
from . import wassdist_utils
from .modules import varphienc4xbar
from .modules import predictorbatchID
from . import wassdist_utils_batchID
#from tqdm.auto import tqdm
from tqdm.notebook import tqdm, trange
import wandb



class InFlowVarDist(nn.Module):
    '''
    Variational distribution for inflow.
    '''
    def __init__(
            self,
            module_genmodel:InFlowGenerativeModel,
            type_impanddisentgl,
            kwargs_impanddisentgl:dict,
            module_varphi_enc_int:nn.Module,
            module_varphi_enc_spl:nn.Module,
            type_cond4flowvarphi0,
            kwargs_cond4flowvarphi0:dict,
            dict_qname_to_scaleandunweighted:dict,
            list_ajdmatpredloss:ListAdjMatPredLoss,
            module_conditionalflowmatcher:utils_flowmatching.ConditionalFlowMatcher,
            coef_P1loss:float,
            module_classifier_P1loss:nn.Module,
            coef_P3loss:float,
            module_predictor_P3loss:nn.Module,
            str_modeP3loss_regorcls:str,
            module_annealing:kl_annealing.AnnealingSchedule,
            weight_logprob_zinbpos:float,
            weight_logprob_zinbzero:float,
            flag_drop_loss_logQdisentangler:bool,
            coef_xbarintCT_loss:float,
            module_classifier_xbarintCT:nn.Module,
            coef_xbarsplNCC_loss:float,
            module_predictor_xbarsplNCC:nn.Module,
            str_modexbarsplNCCloss_regorcls:str,
            coef_rankloss_Z: float,
            module_predictor_ranklossZ_X: nn.Module,
            module_predictor_ranklossZ_Y: nn.Module,
            coef_rankloss_xbarint:float,
            module_predictor_ranklossxbarint_X:nn.Module,
            module_predictor_ranklossxbarint_Y:nn.Module,
            num_subsample_XYrankloss:int,
            coef_xbarint2notNCC_loss: float,
            module_predictor_xbarint2notNCC: nn.Module,
            str_modexbarint2notNCCloss_regorclsorwassdist: str,
            coef_z2notNCC_loss: float,
            module_predictor_z2notNCC: nn.Module,
            str_modez2notNCCloss_regorclsorwassdist: str,
            module_predictor_xbarint2notbatchID:predictorbatchID.PredictorBatchID,
            coef_xbarint2notbatchID_loss:float,
            module_predictor_xbarspl2notbatchID: predictorbatchID.PredictorBatchID,
            coef_xbarspl2notbatchID_loss: float,
    ):
        '''

        :param module_genmodel: the generative model
            TODO: double-check: Important note: in synthetic setting two instances of the generative model are used
                - the one used to generate the observations (not passed here).
                - the one passed here, which has the same architecture to the prev but is has different params.
        :param type_impanddisentgl: either 'disentonly.Disentangler' or `impanddisent.ImputerAndDisentangler`
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
        :param coef_P1loss
        :param flag_drop_loss_logQdisentangler: if set to True, the term logq(x_int, x_spl | x) is dropped from the loss.
        :param coef_xbarintCT_loss: the coeffficient of loss of cell type being predictable from xbar_int,
        :param module_classifier_xbarintCT:nn.Module,
        :param coef_xbarsplNCC_loss: the coefficient of loss of NCC being predictable from xbar_spl.
        :param module_predictor_xbarsplNCC:nn.Module,
        :param str_modexbarsplNCCloss_regorcls:str

        '''
        super(InFlowVarDist, self).__init__()
        # grab args ===
        self.module_genmodel = module_genmodel
        self.type_impanddisentgl = type_impanddisentgl
        self.type_cond4flowvarphi0 = type_cond4flowvarphi0
        self.kwargs_impanddisentgl = kwargs_impanddisentgl
        self.module_varphi_enc_int = module_varphi_enc_int
        self.module_varphi_enc_spl = module_varphi_enc_spl
        self.kwargs_cond4flowvarphi0 = kwargs_cond4flowvarphi0
        self.dict_qname_to_scaleandunweighted = dict_qname_to_scaleandunweighted
        self.list_ajdmatpredloss = list_ajdmatpredloss
        self.module_conditionalflowmatcher = module_conditionalflowmatcher
        self.coef_rankloss_xbarint = coef_rankloss_xbarint
        self.module_predictor_ranklossxbarint_X = module_predictor_ranklossxbarint_X
        self.module_predictor_ranklossxbarint_Y = module_predictor_ranklossxbarint_Y
        self.num_subsample_XYrankloss = num_subsample_XYrankloss
        self.coef_xbarint2notNCC_loss = coef_xbarint2notNCC_loss
        self.module_predictor_xbarint2notNCC = module_predictor_xbarint2notNCC
        self.str_modexbarint2notNCCloss_regorclsorwassdist = str_modexbarint2notNCCloss_regorclsorwassdist


        assert (
            isinstance(module_annealing, kl_annealing.AnnealingSchedule) or (module_annealing is None)
        )
        if module_annealing is not None:
            self.module_annealing = iter(module_annealing)
        else:
            self.module_annealing = None

        self.weight_logprob_zinbpos = weight_logprob_zinbpos
        self.weight_logprob_zinbzero = weight_logprob_zinbzero
        self.flag_drop_loss_logQdisentangler = flag_drop_loss_logQdisentangler
        assert (
            self.flag_drop_loss_logQdisentangler in [True, False]
        )


        # args related to P1 loss
        self.coef_P1loss = coef_P1loss
        self.module_classifier_P1loss = module_classifier_P1loss
        self.crit_P1loss = nn.CrossEntropyLoss()

        # args related to P3 loss
        self.coef_P3loss = coef_P3loss
        self.module_predictor_P3loss = module_predictor_P3loss
        self.str_modeP3loss_regorcls = str_modeP3loss_regorcls
        assert (self.str_modeP3loss_regorcls in ['reg', 'cls'])
        self.crit_P3loss = nn.MSELoss() if(self.str_modeP3loss_regorcls == 'reg') else nn.BCEWithLogitsLoss()

        # related to pred: xbarint --> CT loss
        self.coef_xbarintCT_loss = coef_xbarintCT_loss
        self.module_classifier_xbarintCT = module_classifier_xbarintCT
        self.crit_loss_xbarint2CT = nn.CrossEntropyLoss()

        # related to pred: xbarspl --> NCC loss
        self.coef_xbarsplNCC_loss = coef_xbarsplNCC_loss
        self.module_predictor_xbarsplNCC = module_predictor_xbarsplNCC
        self.str_modexbarsplNCCloss_regorcls = str_modexbarsplNCCloss_regorcls
        assert (self.str_modexbarsplNCCloss_regorcls in ['reg', 'cls'])
        self.crit_loss_xbarspl2NCC = nn.MSELoss() if(self.str_modexbarsplNCCloss_regorcls == 'reg') else nn.BCEWithLogitsLoss()

        # related to xbar_int rank loss ===
        self.crit_xbarint_rankloss = nn.MarginRankingLoss(margin=0.0)

        # related to xbraint 2 not NCC loss ===
        assert (self.str_modexbarint2notNCCloss_regorclsorwassdist in ['reg', 'cls', 'wassdist'])
        if self.str_modexbarint2notNCCloss_regorclsorwassdist in ['reg', 'cls']:
            raise NotImplementedError("reg or cls is replaced with wassdist")
        else:
            self.crit_loss_xbarint2notNCC = wassdist_utils.WassDist(
                coef_fminf=self.coef_xbarint2notNCC_loss,
                coef_smoothness=1.0 * self.coef_xbarint2notNCC_loss
            )


        # related to z 2 not NCC loss ===
        self.coef_z2notNCC_loss = coef_z2notNCC_loss
        self.module_predictor_z2notNCC = module_predictor_z2notNCC
        self.str_modez2notNCCloss_regorclsorwassdist = str_modez2notNCCloss_regorclsorwassdist
        assert (self.str_modez2notNCCloss_regorclsorwassdist in ['reg', 'cls', 'wassdist'])
        if self.str_modez2notNCCloss_regorclsorwassdist in ['reg', 'cls']:
            raise NotImplementedError("reg or cls is replaced by wassdist mode.")
            self.crit_loss_z2notNCC = predictorperCT.MinRowLoss(nn.MSELoss) if (self.str_modez2notNCCloss_regorcls == 'reg') else predictorperCT.MinRowLoss(nn.BCEWithLogitsLoss)
        else:
            self.crit_loss_z2notNCC = wassdist_utils.WassDist(
                coef_fminf=self.coef_z2notNCC_loss,
                coef_smoothness=1.0*self.coef_z2notNCC_loss
            )

        # realted to rank loss for Z
        self.coef_rankloss_Z = coef_rankloss_Z
        self.module_predictor_ranklossZ_X = module_predictor_ranklossZ_X
        self.module_predictor_ranklossZ_Y = module_predictor_ranklossZ_Y
        self.crit_Z_rankloss = nn.MarginRankingLoss(margin=0.0)

        # make internals
        self.module_impanddisentgl = self.type_impanddisentgl(**kwargs_impanddisentgl)
        self.module_cond4flowvarphi0 = type_cond4flowvarphi0(**kwargs_cond4flowvarphi0)

        # related to xbarint2notbatchID
        self.module_predictor_xbarint2notbatchID = module_predictor_xbarint2notbatchID
        self.coef_xbarint2notbatchID_loss = coef_xbarint2notbatchID_loss
        self.crit_loss_xbarint2notbatchID = wassdist_utils_batchID.WassDistBatchID(
            coef_fminf=self.coef_xbarint2notbatchID_loss,
            coef_smoothness=1.0 * self.coef_xbarint2notbatchID_loss
        )

        # related to xbarspl2notbatchID
        self.module_predictor_xbarspl2notbatchID = module_predictor_xbarspl2notbatchID
        self.coef_xbarspl2notbatchID_loss = coef_xbarspl2notbatchID_loss
        self.crit_loss_xbarspl2notbatchID = wassdist_utils_batchID.WassDistBatchID(
            coef_fminf=self.coef_xbarspl2notbatchID_loss,
            coef_smoothness=1.0 * self.coef_xbarspl2notbatchID_loss
        )


        self._check_args()

    def rsample(self, batch, prob_maskknowngenes:float, ten_xy_absolute:torch.Tensor):
        # step 1, rsample from imputer and disentangler
        params_q_impanddisentgl = self.module_impanddisentgl(
            batch=batch,
            prob_maskknowngenes=prob_maskknowngenes,
            ten_xy_absolute=ten_xy_absolute
        )  # so it's not repeated in compgraph.

        if isinstance(self.module_impanddisentgl, GNNDisentangler): # with GNN disentagler, the encoder's varaice is used.
            x_int = Normal(
                loc=params_q_impanddisentgl['muxint'],
                scale=params_q_impanddisentgl['sigmaxint']
            ).rsample().clamp(
                min=torch.zeros_like(params_q_impanddisentgl['x_cnt']),
                max=params_q_impanddisentgl['x_cnt']
            )  # [N, num_genes]
            x_spl = probutils.Normal(
                loc=params_q_impanddisentgl['muxspl'],
                scale=params_q_impanddisentgl['sigmaxspl']
            ).rsample().clamp(
                min=torch.zeros_like(params_q_impanddisentgl['x_cnt']),
                max=params_q_impanddisentgl['x_cnt']
            )  # [N, num_genes]

            if torch.any(torch.isnan(x_int)):
                x_int = torch.nan_to_num(x_int)
                print("Nan Occured in x_int")
                print("     is nan in muxint?: {}".format(
                    torch.any(torch.isnan(params_q_impanddisentgl['muxint']))
                ))
                print("     is nan in sigmaxint?: {}".format(
                    torch.any(torch.isnan(params_q_impanddisentgl['sigmaxint']))
                ))

            if torch.any(torch.isnan(x_spl)):
                x_spl = torch.nan_to_num(x_spl)
                print("Nan Occured in x_spl")
                print("     is nan in muxspl?: {}".format(
                    torch.any(torch.isnan(params_q_impanddisentgl['muxspl']))
                ))
                print("     is nan in sigmaxspl?: {}".format(
                    torch.any(torch.isnan(params_q_impanddisentgl['sigmaxspl']))
                ))

        else:
            raise NotImplementedError(
                "Only the distangler implementation in modules.gnn_disentangler.GNNDisentangler is tested."
            )
            x_int = probutils.ExtenededNormal(
                loc=params_q_impanddisentgl['muxint'],
                scale=self.dict_qname_to_scaleandunweighted['impanddisentgl_int']['scale'],
                flag_unweighted=self.dict_qname_to_scaleandunweighted['impanddisentgl_int']['flag_unweighted']
            ).rsample().clamp(
                min=torch.zeros_like(params_q_impanddisentgl['x_cnt']),
                max=params_q_impanddisentgl['x_cnt']
            )  # [N, num_genes]  # [N, num_genes]
            x_spl = probutils.ExtenededNormal(
                loc=params_q_impanddisentgl['muxspl'],
                scale=self.dict_qname_to_scaleandunweighted['impanddisentgl_spl']['scale'],
                flag_unweighted=self.dict_qname_to_scaleandunweighted['impanddisentgl_spl']['flag_unweighted']
            ).rsample().clamp(
                min=torch.zeros_like(params_q_impanddisentgl['x_cnt']),
                max=params_q_impanddisentgl['x_cnt']
            )  # [N, num_genes]  # [N, num_genes]

        # step 2, rsample from encoders to the low-dim embedding space.
        param_q_xbarint = self.module_varphi_enc_int(x_int, batch)  # [N, dim_latent]
        param_q_xbarspl = self.module_varphi_enc_spl(x_spl, batch)  # [N, dim_latent]
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

        #set ten_u_int and ten_u_spl ===
        assert (
            batch.y.size()[1] == (batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']  + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb'])
        )
        ten_u_int = batch.y[
            :,
            0:batch.INFLOWMETAINF['dim_u_int']
        ].to(ten_xy_absolute.device) if(self.module_genmodel.flag_use_int_u) else None
        ten_u_spl = batch.y[
            :,
            batch.INFLOWMETAINF['dim_u_int']:batch.INFLOWMETAINF['dim_u_int']+batch.INFLOWMETAINF['dim_u_spl']
        ].to(ten_xy_absolute.device) if(self.module_genmodel.flag_use_spl_u) else None


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
            ten_out_imputer=params_q_impanddisentgl['ten_out_imputer'],
            ten_u_int=ten_u_int,
            ten_u_spl=ten_u_spl
        )

    def log_prob(self, dict_retvalrsample):

        if isinstance(self.module_impanddisentgl, GNNDisentangler):
            # use the std-s returned by the disentangler module
            # xint
            logq_xint = Normal(
                loc=dict_retvalrsample['params_q_impanddisentgl']['muxint'],
                scale=dict_retvalrsample['params_q_impanddisentgl']['sigmaxint']
            ).log_prob(dict_retvalrsample['x_int'])  # [N, num_genes]

            # xspl
            logq_xspl = Normal(
                loc=dict_retvalrsample['params_q_impanddisentgl']['muxspl'],
                scale=dict_retvalrsample['params_q_impanddisentgl']['sigmaxspl']
            ).log_prob(dict_retvalrsample['x_spl'])  # [N, num_genes]


        else:
            raise NotImplementedError(
                "The disentangler has to be an instance of inflow.modules.gnn_disentangler import GNNDisentangler"
            )
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
        # TODO: loc should be re-computed with encoder ?
        logq_xbarint = probutils.ExtenededNormal(
            loc=dict_retvalrsample['param_q_xbarint'],
            scale=self.dict_qname_to_scaleandunweighted['varphi_enc_int']['scale'],
            flag_unweighted=self.dict_qname_to_scaleandunweighted['varphi_enc_int']['flag_unweighted']
        ).log_prob(dict_retvalrsample['xbar_int'])  # [N, dim_latent]

        # xbarspl
        # TODO: loc should be re-computed with encoder ?
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

    def train_separately_encdec(self, adata, device, train_fraction:float, val_fraction:float, lr_optim:float, num_epochs:int, str_mode_x:str, batch_size:int):
        '''
        Trains the encoder and decoder (i.e. \theta_{dec} and \varphi_{enc} in the paper), as done in latent diffusion.
        :param adata:
        :param str_mode_x: a string in ['raw', 'log1p']
            - raw: no transformation on x.
            - log1p: log(1+x) is fed to end/dec
        :return:
        '''

        raise NotImplementedError(
            "train_separately_encdec is not implemented when having batch tokens"
        )

        # get x
        x = torch.sparse_coo_tensor(
            indices=adata.X.tocoo().nonzero(),
            values=adata.X.tocoo().data,
            size=adata.X.tocoo().shape
        ).float()  # TODO: could vary based on type(adata.X)? specially the .tocoo part?


        if adata.X.shape[0] * adata.X.shape[1] < 1e9:
            assert (
                torch.all(
                    torch.isclose(
                        x.to_dense().float(),
                        torch.tensor(adata.X.toarray()).float()
                    )
                )
            )
            print("assertion was passed.")

        x = x.to_dense()  # since TensorDataset couldn't handle sparse_coo_tensor

        if str_mode_x == 'raw':
            pass
        elif str_mode_x == 'log1p':
            x = torch.log(1+x)

        # split the ds to train/val/test
        assert (isinstance(train_fraction, float) and (train_fraction > 0.0) and (train_fraction < 1.0))
        assert (isinstance(val_fraction, float) and (val_fraction > 0.0) and (val_fraction < 1.0))
        N_train = int(train_fraction * adata.X.shape[0])
        N_val   = int(val_fraction * adata.X.shape[0])
        N_test  = adata.X.shape[0] - N_val - N_train
        idx_rnd_perm = np.random.permutation(adata.X.shape[0]).tolist()
        list_idx_train = idx_rnd_perm[0:N_train] + []
        list_idx_val   = idx_rnd_perm[N_train:N_train+N_val]
        list_idx_test  = idx_rnd_perm[N_train+N_val::]
        assert (
            set(list_idx_train).intersection(set(list_idx_val)) == set([])
        )
        assert (
            set(list_idx_train).intersection(set(list_idx_test)) == set([])
        )
        assert (
            set(list_idx_val).intersection(set(list_idx_test)) == set([])
        )
        assert (
            set(list_idx_train).union(set(list_idx_val)).union(set(list_idx_test)) == set(range(adata.X.shape[0]))
        )
        ds_train = torch.utils.data.TensorDataset(x[list_idx_train, :] + 0.0)
        ds_val   = torch.utils.data.TensorDataset(x[list_idx_val, :] + 0.0)
        ds_test  = torch.utils.data.TensorDataset(x[list_idx_test, :] + 0.0)
        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, num_workers=4)
        dl_val   = torch.utils.data.DataLoader(ds_val, batch_size=batch_size,  num_workers=4, shuffle=False)
        dl_test  = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=4, shuffle=False)


        # check if `module_varphi_enc_int` and `module_varphi_enc_spl` are two separate modules, are the same module
        flag_varphiint_sameas_varphispl = list(self.module_varphi_enc_int.parameters())[0].data_ptr() == list(self.module_varphi_enc_spl.parameters())[0].data_ptr()
        flag_thetaint_sameas_thetaspl = list(self.module_genmodel.module_w_dec_int.parameters())[0].data_ptr() == list(self.module_genmodel.module_w_dec_spl.parameters())[0].data_ptr()

        if not flag_varphiint_sameas_varphispl:
            raise NotImplementedError('"At least in this function" module_varphi_enc_int and module_varphi_enc_int are assumed the same.')

        if not flag_thetaint_sameas_thetaspl:
            raise NotImplementedError('"At least in this function" w_dec_int and w_dec_spl are assumed the same.')


        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.module_varphi_enc_int.parameters() if(flag_varphiint_sameas_varphispl) else list(self.module_varphi_enc_int.parameters()) + list(self.module_varphi_enc_spl.parameters()),
            lr=lr_optim
        )

        hist_loss_train, hist_loss_validation, hist_loss_test =[], [], []
        for idx_epoch in tqdm(range(num_epochs)):
            # train
            for _, data in enumerate(dl_train):
                assert (len(data) == 1)
                optimizer.zero_grad()
                netout = self.module_genmodel.module_w_dec_int(
                    self.module_varphi_enc_int(data[0].to(device))
                )
                loss = criterion(netout, data[0].to(device))
                loss.backward()
                optimizer.step()
                hist_loss_train.append(loss.detach().cpu().numpy().tolist())

            # validation
            list_tmp = []
            with torch.no_grad():
                for _, data in enumerate(dl_val):
                    assert (len(data) == 1)
                    list_tmp.append(
                        criterion(
                            self.module_genmodel.module_w_dec_int(
                                self.module_varphi_enc_int(data[0].to(device))
                            ),
                            data[0].to(device)
                        ).detach().cpu().numpy().tolist()
                    )
            hist_loss_validation.append(
                [len(hist_loss_train), np.mean(list_tmp)]
            )

            # test
            list_tmp = []
            with torch.no_grad():
                for _, data in enumerate(dl_test):
                    assert (len(data) == 1)
                    list_tmp.append(
                        criterion(
                            self.module_genmodel.module_w_dec_int(
                                self.module_varphi_enc_int(data[0].to(device))
                            ),
                            data[0].to(device)
                        ).detach().cpu().numpy().tolist()
                    )
            hist_loss_test.append(
                [len(hist_loss_train), np.mean(list_tmp)]
            )

        return hist_loss_train, hist_loss_validation, hist_loss_test






    def training_epoch(
            self,
            dl:NeighborLoader | List[NeighborLoader],
            prob_maskknowngenes:float,
            t_num_steps:int,
            ten_xy_absolute:torch.Tensor | List[torch.Tensor],
            optim_training:torch.optim.Optimizer,
            tensorboard_stepsize_save:int,
            prob_applytfm_affinexy:float,
            np_size_factor: np.ndarray | List[np.ndarray],
            flag_lockencdec_duringtraining,
            numsteps_accumgrad:int,
            num_updateseparate_afterGRLs:int,
            itrcount_wandbstep_input:int|None=None,
            list_flag_elboloss_imputationloss=[True, True],
            coef_loss_closeness_zz_xbarintxbarint_xintxint:float=0.0,
            coef_flowmatchingloss:float=0.0,
            flag_verbose:bool=False
    ):
        '''
        One epoch of the training.
        :param flag_lockencdec_duringtraining: if set to True, the encoder/decoder moduels are kept frozen. Because they are assumed to be trained separately (as done in latent diffusion).
            Since the optimizer if fed as an arg, `flag_lockencdec_duringtraining` is only used to check if the frozen parameters are not in optimizer's list.
        :param dl: pyg's neighborloader of a list of neighbour loaders.
        :param prob_maskknowngenes: probability of masking some known gene expressions (per-cell)
            to define the self-supervised loss.
        :param t_num_steps: the number of time-steps to be used by the NeuralODE module.
        :param ten_xy_absolute: absoliute xy positions of all cells.
        :param optim_training: the optimizer.
        :param tensorboard_stepsize_save
        :param itrcount_wandbstep_input
        :param list_flag_elboloss_imputationloss
        :param coef_loss_closeness_zz_xbarintxbarint_xintxint
        :param prob_applytfm_affinexy: with this probability the [xy] positions go throug an affined transformation.
        :param coef_flowmatchingloss: the coefficient for flow-matching loss.
        :param np_size_factor: a tensor of shape [num_cells], containing the size factors.
        :param numsteps_accumgrad: number of gradient accumulation step (to increase the batch-size).
        :param num_updateseparate_afterGRLs
        :return:
        '''

        if isinstance(dl, list):
            assert isinstance(ten_xy_absolute, list)
            assert isinstance(np_size_factor, list)
            for u in dl:
                assert isinstance(u, NeighborLoader)
            for u in ten_xy_absolute:
                assert isinstance(u, torch.Tensor)
            for u in np_size_factor:
                assert isinstance(u, np.ndarray)
            list_dl = dl
            list_ten_xy_absolute = ten_xy_absolute
            list_np_size_fator = np_size_factor
        else:
            assert isinstance(dl, NeighborLoader)
            assert isinstance(ten_xy_absolute, torch.Tensor)
            assert isinstance(np_size_factor, np.ndarray)

            list_dl = [dl]  # to support the old-style calls with `dl` arg being a single dataloader.
            list_ten_xy_absolute = [ten_xy_absolute]  # to support the old-style calls with `dl` arg being a single dataloader.
            list_np_size_fator = [np_size_factor]  # to support the old-style calls with `dl` arg being a single dataloader.

        if self.module_annealing is not None:
            self.coef_anneal = next(self.module_annealing)
        else:
            self.coef_anneal = None



        if flag_lockencdec_duringtraining:
            assert (optim_training.flag_freezeencdec)  # TODO: make the check differently

        # make the affine xy augmenter
        tfm_affinexy = utils_imputer.RandomGeometricTfm(
            prob_applytfm=prob_applytfm_affinexy,
            rng_00=[1.0, 2.0],
            rng_01=[1.0, 2.0],
            rng_10=[1.0, 2.0],
            rng_11=[1.0, 2.0]
        )  # TODO: maybe tune the ranges?


        if itrcount_wandbstep_input is not None:
            itrcount_wandb = itrcount_wandbstep_input + 0
        else:
            itrcount_wandb = 0

        list_coef_anneal = []

        num_backwards = 0

        # iterpygdl_for_afterGRL = iter(dl)

        list_iter_dl_normal = [iter(u) for u in list_dl]

        list_iter_dl_2ndpygbatch = [iter(u) for u in list_dl]  # the 2nd pygbatch for batch mixing (i.e. xbarint/xbarspl --> notbatchID losses)

        list_iterfinished_normal = [False for u in list_dl]

        idx_current_dl_normal = -1


        optim_training.zero_grad()
        temp_print_whilecount = -1
        while not np.all(list_iterfinished_normal): # for batch in tqdm(dl):
            temp_print_whilecount += 1
            idx_current_dl_normal = (idx_current_dl_normal + 1)%len(list_dl)
            try:
                itr = list_iter_dl_normal[idx_current_dl_normal]
                batch = next(itr)
            except StopIteration:
                if flag_verbose:
                    print("<><><><><><><><><><><><><><><><><> stop-iteration occured at while-loop {}".format(temp_print_whilecount))

                list_iterfinished_normal[idx_current_dl_normal] = True
                list_iter_dl_normal[idx_current_dl_normal] = iter(list_dl[idx_current_dl_normal])
                itr = list_iter_dl_normal[idx_current_dl_normal]
                batch = next(itr)

            if np.all(list_iterfinished_normal):
                break

            wandb.log(
                {"InspectVals/annealing_coefficient": self.coef_anneal},
                step=itrcount_wandb
            )

            batch.INFLOWMETAINF = {
                "dim_u_int": self.module_genmodel.dict_varname_to_dim['u_int'],
                "dim_u_spl": self.module_genmodel.dict_varname_to_dim['u_spl'],
                "dim_CT":self.module_genmodel.dict_varname_to_dim['CT'],
                "dim_NCC":self.module_genmodel.dict_varname_to_dim['NCC'],
                "dim_BatchEmb":self.module_genmodel.dict_varname_to_dim['BatchEmb']
            }  # how batch.y is split between u_int, u_spl, CT, NCC, BatchEmb

            ten_xy_touse = list_ten_xy_absolute[idx_current_dl_normal]

            self.module_genmodel.clamp_thetanegbins()

            # optim_training.zero_grad()
            flag_tensorboardsave = (itrcount_wandb%tensorboard_stepsize_save == 0)

            dict_q_sample = self.rsample(
                batch=batch,
                prob_maskknowngenes=prob_maskknowngenes,
                ten_xy_absolute=ten_xy_touse
            )
            dict_logq = self.log_prob(dict_q_sample)
            dict_logp, dict_otherinf = self.module_genmodel.log_prob(
                dict_qsamples=dict_q_sample,
                batch=batch,
                t_num_steps=t_num_steps,
                np_size_factor=list_np_size_fator[idx_current_dl_normal],
                coef_anneal=self.coef_anneal
            )
            list_coef_anneal.append(dict_otherinf['coef_anneal'])


            # make the loss
            loss = 0.0
            if list_flag_elboloss_imputationloss[0]:
                for k in dict_logp.keys():
                    if k not in ['logp_x_int', 'logp_x_spl']:
                        lossterm_logp = dict_logp[k].sum(1).mean()
                        loss = loss - lossterm_logp
                    else:
                        assert (k in ['logp_x_int', 'logp_x_spl'])

                        if self.weight_logprob_zinbpos == -1:  # do as usual
                            assert (self.weight_logprob_zinbzero == -1)
                            lossterm_logp = dict_logp[k].sum(1).mean()
                            loss = loss - lossterm_logp
                        else:
                            x_cnt = batch.x.to_dense().to(list_ten_xy_absolute[0].device).detach()[:batch.batch_size] + 0.0
                            lossterm_logp_pos = dict_logp[k][x_cnt > 0]
                            lossterm_logp_zero = dict_logp[k][x_cnt == 0]
                            lossterm_logp = self.weight_logprob_zinbpos*lossterm_logp_pos.sum() + self.weight_logprob_zinbzero*lossterm_logp_zero.sum()
                            lossterm_logp = lossterm_logp/((x_cnt.size()[0] + 0.0) * (self.weight_logprob_zinbpos + self.weight_logprob_zinbzero))
                            loss = loss - lossterm_logp


                    if flag_tensorboardsave:
                        with torch.no_grad():
                            wandb.log(
                                {"Loss/logprob_P/{}".format(k): -lossterm_logp},
                                step=itrcount_wandb
                            )

            if list_flag_elboloss_imputationloss[0]:
                for k in dict_logq.keys():
                    if k not in ['logq_xint', 'logq_xspl']:
                        lossterm_logq = dict_logq[k].sum(1).mean()
                        loss = loss + lossterm_logq
                    else:
                        if not self.flag_drop_loss_logQdisentangler:
                            # q(x_int|x) and q(x_spl|x) handled on non-zero elements.
                            assert k in ['logq_xint', 'logq_xspl']

                            x_cnt = batch.x.to_dense().to(list_ten_xy_absolute[0].device).detach() + 0.0
                            lossterm_logq = (dict_logq[k][x_cnt > 0].sum())/(x_cnt.size()[0]+0.0)
                            loss = loss + lossterm_logq
                        else:
                            lossterm_logq = -1

                    if flag_tensorboardsave:
                        with torch.no_grad():
                            wandb.log(
                                {"Loss/logprob_Q/{}".format(k): lossterm_logq},
                                step=itrcount_wandb
                            )


            # add the loss terms to predict adjecancy matrix ===
            if len(self.list_ajdmatpredloss.list_adjpredictors) > 0:
                dict_varname_to_adjpredloss = self.list_ajdmatpredloss(
                    dict_q_sample=dict_q_sample,
                    pyg_batch=batch
                )
                for varname in dict_varname_to_adjpredloss.keys():
                    loss = loss + dict_varname_to_adjpredloss[varname]

                if flag_tensorboardsave:
                    with torch.no_grad():
                        for varname in dict_varname_to_adjpredloss.keys():
                            wandb.log(
                                {"Loss_PredAdj/{}".format(varname): dict_varname_to_adjpredloss[varname]},
                                step=itrcount_wandb
                            )


            # add the flow-matching loss ===
            if coef_flowmatchingloss > 0.0:
                rng_batchemb = [
                    batch.INFLOWMETAINF['dim_u_int']+batch.INFLOWMETAINF['dim_u_spl']+batch.INFLOWMETAINF['dim_CT']+batch.INFLOWMETAINF['dim_NCC'],
                    batch.INFLOWMETAINF['dim_u_int']+batch.INFLOWMETAINF['dim_u_spl']+batch.INFLOWMETAINF['dim_CT']+batch.INFLOWMETAINF['dim_NCC']+batch.INFLOWMETAINF['dim_BatchEmb']
                ]

                fm_loss = self.module_conditionalflowmatcher.get_fmloss(
                    module_v=self.module_genmodel.module_Vflow_unwrapped,
                    x1=torch.cat(
                    [
                        dict_q_sample['xbar_int'][:batch.batch_size],
                        dict_q_sample['xbar_spl'][:batch.batch_size]
                    ],
                    1
                    ),
                    x0_frominflow=torch.cat(
                    [
                        dict_q_sample['z'][:batch.batch_size],
                        dict_q_sample['s_in'][:batch.batch_size]
                    ],
                    1
                    ),
                    ten_batchEmb=batch.y[
                        :,
                        rng_batchemb[0]:rng_batchemb[1]
                    ][:batch.batch_size].to(list_ten_xy_absolute[0].device)
                )
                loss = loss + coef_flowmatchingloss*fm_loss

                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/FMloss (after mult by coef={})".format(coef_flowmatchingloss): coef_flowmatchingloss*fm_loss},
                            step=itrcount_wandb
                        )

            # add P1 loss ===
            if self.coef_P1loss > 0.0:
                rng_CT = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
                ]
                P1loss = self.crit_P1loss(
                    self.module_classifier_P1loss(
                        dict_q_sample['param_q_cond4flow']['mu_z']
                    ),
                    torch.argmax(
                        batch.y[:, rng_CT[0]:rng_CT[1]].to(list_ten_xy_absolute[0].device),
                        1
                    )
                )
                loss = loss + self.coef_P1loss * P1loss
                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/P1Loss (after mult by coef={})".format(self.coef_P1loss): self.coef_P1loss * P1loss},
                            step=itrcount_wandb
                        )


            # add P3 loss ===
            if self.coef_P3loss > 0.0:
                rng_NCC = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC']
                ]
                ten_NCC = batch.y[
                    :,
                    rng_NCC[0]:rng_NCC[1]
                ].to(list_ten_xy_absolute[0].device).float()

                if self.str_modeP3loss_regorcls == 'cls':
                    ten_NCC = ((ten_NCC > 0) + 0).float()
                else:
                    assert (self.str_modeP3loss_regorcls == 'reg')

                P3loss = self.crit_P3loss(
                    self.module_predictor_P3loss(dict_q_sample['param_q_cond4flow']['mu_sin']),
                    ten_NCC
                )
                loss = loss + self.coef_P3loss * P3loss
                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/P3loss (after mult by coef={})".format(self.coef_P3loss): self.coef_P3loss * P3loss},
                            step=itrcount_wandb
                        )


            # add xbarint-->CT loss ===
            if self.coef_xbarintCT_loss > 0.0:
                rng_CT = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
                ]
                xbarint2CT_loss = self.crit_loss_xbarint2CT(
                    self.module_classifier_xbarintCT(
                        dict_q_sample['xbar_int'][:batch.batch_size]
                    ),
                    torch.argmax(
                        batch.y[:batch.batch_size, rng_CT[0]:rng_CT[1]].to(list_ten_xy_absolute[0].device),
                        1
                    )
                )
                loss = loss + self.coef_xbarintCT_loss * xbarint2CT_loss
                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/xbarint-->CT (after mult by coef={})".format(self.coef_xbarintCT_loss): self.coef_xbarintCT_loss * xbarint2CT_loss},
                            step=itrcount_wandb
                        )


            # add xbarspl-->NCC loss ===
            if self.coef_xbarsplNCC_loss > 0.0:
                rng_NCC = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC']
                ]
                ten_NCC = batch.y[
                    :batch.batch_size,
                    rng_NCC[0]:rng_NCC[1]
                ].to(list_ten_xy_absolute[0].device).float()

                if self.str_modexbarsplNCCloss_regorcls == 'cls':
                    ten_NCC = ((ten_NCC > 0) + 0).float()
                else:
                    assert (self.str_modexbarsplNCCloss_regorcls == 'reg')

                xbarspl2NCC_loss = self.crit_loss_xbarspl2NCC(
                    self.module_predictor_xbarsplNCC(dict_q_sample['xbar_spl'][:batch.batch_size]),
                    ten_NCC
                )
                loss = loss + self.coef_xbarsplNCC_loss * xbarspl2NCC_loss
                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/xbarspl-->NCC (after mult by coef={})".format(self.coef_xbarsplNCC_loss): self.coef_xbarsplNCC_loss * xbarspl2NCC_loss},
                            step=itrcount_wandb
                        )


            # add Z rank loss  ===
            if self.coef_rankloss_Z > 0.0:
                # print("batch.input_id.shape = {}".format(batch.input_id.shape))
                # print("dict_q_sample['xbar_spl'].shape = {}".format(dict_q_sample['xbar_spl'].shape))
                assert (batch.n_id.shape[0] == dict_q_sample['xbar_spl'].shape[0])
                assert (list_ten_xy_absolute[idx_current_dl_normal].size()[1] == 2)
                ten_x, ten_y = list_ten_xy_absolute[idx_current_dl_normal][batch.input_id.tolist(), 0].detach(), list_ten_xy_absolute[idx_current_dl_normal][batch.input_id.tolist(), 1].detach()  # [N], [N]

                # subsample the mini-batch to define the rank loss
                rng_N = tuple(range(ten_x.size()[0]))
                list_ij_subsample = random.sample(
                    [(i, j) for i in rng_N for j in set(rng_N) - {i}],
                    k=min(self.num_subsample_XYrankloss, ten_x.size()[0])
                )
                list_i_subsample = [u[0] for u in list_ij_subsample]
                list_j_subsample = [u[1] for u in list_ij_subsample]

                netout_rank_Xpos = self.module_predictor_ranklossZ_X(
                    predadjmat.grad_reverse(
                        torch.cat(
                            [
                                dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_i_subsample, :],
                                dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_j_subsample, :]
                            ],
                            1
                        )
                    )
                )  # [N,2]  # TODO: should it be on non-cental nodes as well?

                assert (netout_rank_Xpos.size()[1] == 2)
                netout_rank_Ypos = self.module_predictor_ranklossZ_Y(
                    predadjmat.grad_reverse(
                        torch.cat(
                            [
                                dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_i_subsample, :],
                                dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_j_subsample, :]
                            ],
                            1
                        )
                    )
                )  # [N,2]  # TODO: should it be on non-cental nodes as well?
                assert (netout_rank_Ypos.size()[1] == 2)

                loss_rank_Xpos = self.crit_Z_rankloss(
                    netout_rank_Xpos[:, 0],
                    netout_rank_Xpos[:, 1],
                    (ten_x[list_i_subsample] - ten_x[list_j_subsample]).sign()
                )
                loss_rank_Ypos = self.crit_Z_rankloss(
                    netout_rank_Ypos[:, 0],
                    netout_rank_Ypos[:, 1],
                    (ten_y[list_i_subsample] - ten_y[list_j_subsample]).sign()
                )

                loss_rank_XYpos = loss_rank_Xpos + loss_rank_Ypos

                loss = loss + self.coef_rankloss_Z * loss_rank_XYpos
                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/RankXY loss, Z (after mult by coef={})".format(self.coef_rankloss_Z): self.coef_rankloss_Z * loss_rank_XYpos},
                            step=itrcount_wandb
                        )




            # add xbarint rank loss  ===
            if self.coef_rankloss_xbarint > 0.0:

                #print("batch.input_id.shape = {}".format(batch.input_id.shape))
                #print("dict_q_sample['xbar_spl'].shape = {}".format(dict_q_sample['xbar_spl'].shape))
                assert(batch.n_id.shape[0] == dict_q_sample['xbar_spl'].shape[0])
                assert (list_ten_xy_absolute[idx_current_dl_normal].size()[1] == 2)
                ten_x, ten_y = list_ten_xy_absolute[idx_current_dl_normal][batch.input_id.tolist(), 0].detach(), list_ten_xy_absolute[idx_current_dl_normal][batch.input_id.tolist(), 1].detach()  # [N], [N]

                # subsample the mini-batch to define the rank loss
                rng_N = tuple(range(ten_x.size()[0]))
                list_ij_subsample = random.sample(
                    [(i, j) for i in rng_N for j in set(rng_N)-{i}],
                    k=min(self.num_subsample_XYrankloss, ten_x.size()[0])
                )
                list_i_subsample = [u[0] for u in list_ij_subsample]
                list_j_subsample = [u[1] for u in list_ij_subsample]


                netout_rank_Xpos = self.module_predictor_ranklossxbarint_X(
                    predadjmat.grad_reverse(
                        torch.cat(
                            [
                                dict_q_sample['xbar_int'][:batch.batch_size][list_i_subsample, :],
                                dict_q_sample['xbar_int'][:batch.batch_size][list_j_subsample, :]
                            ],
                            1
                        )
                    )
                )  # [N,2]
                assert (netout_rank_Xpos.size()[1] == 2)
                netout_rank_Ypos = self.module_predictor_ranklossxbarint_Y(
                    predadjmat.grad_reverse(
                        torch.cat(
                            [
                                dict_q_sample['xbar_int'][:batch.batch_size][list_i_subsample, :],
                                dict_q_sample['xbar_int'][:batch.batch_size][list_j_subsample, :]
                            ],
                            1
                        )
                    )
                )  # [N,2]
                assert (netout_rank_Ypos.size()[1] == 2)

                loss_rank_Xpos = self.crit_xbarint_rankloss(
                    netout_rank_Xpos[:, 0],
                    netout_rank_Xpos[:, 1],
                    (ten_x[list_i_subsample] - ten_x[list_j_subsample]).sign()
                )
                loss_rank_Ypos = self.crit_xbarint_rankloss(
                    netout_rank_Ypos[:, 0],
                    netout_rank_Ypos[:, 1],
                    (ten_y[list_i_subsample] - ten_y[list_j_subsample]).sign()
                )

                loss_rank_XYpos = loss_rank_Xpos + loss_rank_Ypos


                loss = loss + self.coef_rankloss_xbarint * loss_rank_XYpos
                if flag_tensorboardsave:
                    with torch.no_grad():
                        wandb.log(
                            {"Loss/RankXY loss, xbarint (after mult by coef={})".format(self.coef_rankloss_xbarint): self.coef_rankloss_xbarint * loss_rank_XYpos},
                            step=itrcount_wandb
                        )

            # add xbarint-->notNCC loss ===
            if self.coef_xbarint2notNCC_loss > 0.0:
                rng_CT = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
                ]

                rng_NCC = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC']
                ]

                ten_NCC = batch.y[
                    :,
                    rng_NCC[0]:rng_NCC[1]
                ].to(list_ten_xy_absolute[0].device).float()

                if self.str_modexbarint2notNCCloss_regorclsorwassdist in ['cls', 'wassdist']:
                    ten_NCC = ((ten_NCC > 0) + 0).float()
                else:
                    raise NotImplementedError("ddd")
                    assert (self.str_modexbarint2notNCCloss_regorclsorwassdist == 'reg')

                dict_xbarint2notNCC_loss = self.crit_loss_xbarint2notNCC(
                    z=predadjmat.grad_reverse(
                        dict_q_sample['param_q_xbarint']
                    ),
                    module_NCCpredictor=self.module_predictor_xbarint2notNCC,
                    ten_CT=batch.y[:, rng_CT[0]:rng_CT[1]],
                    ten_NCC=ten_NCC.detach()
                )


                for loss_name in dict_xbarint2notNCC_loss.keys():
                    loss = loss + dict_xbarint2notNCC_loss[loss_name]['coef'] * dict_xbarint2notNCC_loss[loss_name]['val']

                if flag_tensorboardsave:
                    with torch.no_grad():
                        for loss_name in dict_xbarint2notNCC_loss.keys():
                            wandb.log(
                                {"Loss/xbarint-->notNCC {} (after mult by coef={})".format(loss_name, dict_xbarint2notNCC_loss[loss_name]['coef']):dict_xbarint2notNCC_loss[loss_name]['coef'] * dict_xbarint2notNCC_loss[loss_name]['val']},
                                step=itrcount_wandb
                            )

            # add Z-->notNCC loss ===
            if self.coef_z2notNCC_loss > 0.0:
                rng_CT = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
                ]

                rng_NCC = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC']
                ]

                ten_NCC = batch.y[
                    :,
                    rng_NCC[0]:rng_NCC[1]
                ].to(list_ten_xy_absolute[0].device).float()

                if self.str_modez2notNCCloss_regorclsorwassdist in ['cls', 'wassdist']:
                    ten_NCC = ((ten_NCC > 0) + 0).float()
                else:
                    raise NotImplementedError("ddd")
                    assert (self.str_modez2notNCCloss_regorclsorwassdist == 'reg')

                dict_z2notNCC_loss = self.crit_loss_z2notNCC(
                    z=predadjmat.grad_reverse(
                        dict_q_sample['param_q_cond4flow']['mu_z']
                    ),
                    module_NCCpredictor=self.module_predictor_z2notNCC,
                    ten_CT=batch.y[:, rng_CT[0]:rng_CT[1]],
                    ten_NCC=ten_NCC.detach()
                )


                for loss_name in dict_z2notNCC_loss.keys():
                    loss = loss + dict_z2notNCC_loss[loss_name]['coef'] * dict_z2notNCC_loss[loss_name]['val']


                if flag_tensorboardsave:
                    with torch.no_grad():
                        for loss_name in dict_z2notNCC_loss.keys():
                            wandb.log(
                                {"Loss/Z-->notNCC {} (after mult by coef={})".format(loss_name, dict_z2notNCC_loss[loss_name]['coef']): dict_z2notNCC_loss[loss_name]['coef'] * dict_z2notNCC_loss[loss_name]['val']},
                                step=itrcount_wandb
                            )


            # get a 2nd pygdl and q_samples for the below two losses
            try:
                batch_2ndpygbatch = next(list_iter_dl_2ndpygbatch[(idx_current_dl_normal + 1)%len(list_dl)])
            except StopIteration:
                list_iter_dl_2ndpygbatch[(idx_current_dl_normal + 1) % len(list_dl)] = iter(list_dl[(idx_current_dl_normal + 1)%len(list_dl)])
                batch_2ndpygbatch = next(list_iter_dl_2ndpygbatch[(idx_current_dl_normal + 1) % len(list_dl)])

            batch_2ndpygbatch.INFLOWMETAINF = batch.INFLOWMETAINF

            dict_q_sample_2ndpygbatch = self.rsample(
                batch=batch_2ndpygbatch,
                prob_maskknowngenes=prob_maskknowngenes,
                ten_xy_absolute=list_ten_xy_absolute[(idx_current_dl_normal + 1) % len(list_dl)]
            )

            # add xbarint-->notBatchID loss ===
            if self.coef_xbarint2notbatchID_loss > 0.0:

                rng_batchemb = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb']
                ]


                dict_xbarint2notbatchID_loss = self.crit_loss_xbarint2notbatchID(
                    z=predadjmat.grad_reverse(
                        torch.cat(
                            [dict_q_sample['param_q_xbarint'], dict_q_sample_2ndpygbatch['param_q_xbarint']],
                            0
                        )
                    ),
                    module_BatchIDpredictor=self.module_predictor_xbarint2notbatchID,
                    ten_BatchID=torch.cat(
                        [batch.y[:, rng_batchemb[0]:rng_batchemb[1]], batch_2ndpygbatch.y[:, rng_batchemb[0]:rng_batchemb[1]]],
                        0
                    ).to(list_ten_xy_absolute[0].device)
                )

                for loss_name in dict_xbarint2notbatchID_loss.keys():
                    loss = loss + dict_xbarint2notbatchID_loss[loss_name]['coef'] * dict_xbarint2notbatchID_loss[loss_name]['val']

                if flag_tensorboardsave:
                    with torch.no_grad():
                        for loss_name in dict_xbarint2notbatchID_loss.keys():
                            wandb.log(
                                {"Loss/xbarint-->notBatchID {} (after mult by coef={})".format(loss_name, dict_xbarint2notbatchID_loss[loss_name]['coef']):
                                  dict_xbarint2notbatchID_loss[loss_name]['coef'] * dict_xbarint2notbatchID_loss[loss_name]['val']
                                },
                                step=itrcount_wandb
                            )

            # add xbarspl-->notBatchID loss ===
            if self.coef_xbarspl2notbatchID_loss > 0.0:

                rng_batchemb = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb']
                ]

                dict_xbarspl2notbatchID_loss = self.crit_loss_xbarspl2notbatchID(
                    z=predadjmat.grad_reverse(
                        torch.cat(
                            [dict_q_sample['param_q_xbarspl'], dict_q_sample_2ndpygbatch['param_q_xbarspl']],
                            0
                        )
                    ),
                    module_BatchIDpredictor=self.module_predictor_xbarspl2notbatchID,
                    ten_BatchID=torch.cat(
                        [batch.y[:, rng_batchemb[0]:rng_batchemb[1]], batch_2ndpygbatch.y[:, rng_batchemb[0]:rng_batchemb[1]]],
                        0
                    )
                )#.to(list_ten_xy_absolute[0].device)

                for loss_name in dict_xbarspl2notbatchID_loss.keys():
                    loss = loss + dict_xbarspl2notbatchID_loss[loss_name]['coef'] * dict_xbarspl2notbatchID_loss[loss_name]['val']

                if flag_tensorboardsave:
                    with torch.no_grad():
                        for loss_name in dict_xbarspl2notbatchID_loss.keys():
                            wandb.log(
                                {"Loss/xbarspl-->notBatchID {} (after mult by coef={})".format(loss_name, dict_xbarspl2notbatchID_loss[loss_name]['coef']):
                                     dict_xbarspl2notbatchID_loss[loss_name]['coef'] * dict_xbarspl2notbatchID_loss[loss_name]['val']
                                 },
                                step=itrcount_wandb
                            )




            # log int_cov_u and spl_cov_u ===
            if flag_tensorboardsave:
                with torch.no_grad():
                    # int_cov_u
                    if dict_otherinf['int_cov_u'] is not None:
                        if isinstance(dict_otherinf['int_cov_u'], float):
                            min_int_cov_u = dict_otherinf['int_cov_u']
                            max_int_cov_u = dict_otherinf['int_cov_u']
                            mean_int_cov_u = dict_otherinf['int_cov_u']
                        elif isinstance(dict_otherinf['int_cov_u'], torch.Tensor):
                            min_int_cov_u = dict_otherinf['int_cov_u'].min()
                            max_int_cov_u = dict_otherinf['int_cov_u'].max()
                            mean_int_cov_u = dict_otherinf['int_cov_u'].mean()
                        else:
                            raise Exception("Unknown type {} for int_cov_u".format(type(
                                dict_otherinf['int_cov_u']
                            )))
                        wandb.log(
                            {"InspectVals/int_cov_u.min":  min_int_cov_u,
                             "InspectVals/int_cov_u.max":  max_int_cov_u,
                             "InspectVals/int_cov_u.mean": mean_int_cov_u},
                            step=itrcount_wandb
                        )

                    # spl_cov_u
                    if dict_otherinf['spl_cov_u'] is not None:
                        if isinstance(dict_otherinf['spl_cov_u'], float):
                            min_spl_cov_u = dict_otherinf['spl_cov_u']
                            max_spl_cov_u = dict_otherinf['spl_cov_u']
                            mean_spl_cov_u = dict_otherinf['spl_cov_u']
                        elif isinstance(dict_otherinf['spl_cov_u'], torch.Tensor):
                            min_spl_cov_u = dict_otherinf['spl_cov_u'].min()
                            max_spl_cov_u = dict_otherinf['spl_cov_u'].max()
                            mean_spl_cov_u = dict_otherinf['spl_cov_u'].mean()
                        else:
                            raise Exception("Unknown type {} for spl_cov_u".format(type(
                                dict_otherinf['spl_cov_u']
                            )))
                        wandb.log(
                            {"InspectVals/spl_cov_u.min": min_spl_cov_u,
                             "InspectVals/spl_cov_u.max": max_spl_cov_u,
                             "InspectVals/spl_cov_u.mean": mean_spl_cov_u},
                            step=itrcount_wandb
                        )


            # log params_q_impanddisentgl['sigmaxint'] and params_q_impanddisentgl['sigmaxint']
            if flag_tensorboardsave and isinstance(self.module_impanddisentgl, GNNDisentangler):
                with torch.no_grad():
                    wandb.log(
                        {"InspectVals/q_disentangler.min": dict_q_sample['params_q_impanddisentgl']['sigmaxint'].min(),
                         "InspectVals/q_disentangler.max": dict_q_sample['params_q_impanddisentgl']['sigmaxint'].max(),
                         "InspectVals/q_disentangler.mean": dict_q_sample['params_q_impanddisentgl']['sigmaxint'].mean()},
                        step=itrcount_wandb
                    )

            # add the z-z, xbarint-xbarint, and xint-xint closeness loss ===
            num_celltypes = self.module_genmodel.dict_varname_to_dim['cell-types']
            if coef_loss_closeness_zz_xbarintxbarint_xintxint > 0.0:
                coef_loss_zzcloseness = coef_loss_closeness_zz_xbarintxbarint_xintxint + 0.0
                rng_CT = [
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                    batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
                ]


                np_celltype_batch = torch.argmax(
                    batch.y[:, rng_CT[0]:rng_CT[1]],
                    1
                ).detach().cpu().numpy().flatten()


                set_celltype_minibatch = list(set(np_celltype_batch.tolist()))
                set_celltype_minibatch.sort()

                MODE_COMP_PAIRWISEDIST = 'pytorchcdist'
                assert MODE_COMP_PAIRWISEDIST in ['pytorchcdist', 'vanila']

                for varname in ['z', 'x_int', 'xbar_int']:
                    loss_zzcloseness = 0.0
                    for ct in set_celltype_minibatch:
                        if np.sum(np_celltype_batch == ct) >= 2:  # if there are atleast two cells of that type.
                            z_incelltype = dict_q_sample[varname][np_celltype_batch == ct, :] + 0.0  # [n, dimz]


                            if varname == 'x_int':  # for x_int the counts are in the order of 1e4 --> defined the closeness loss on log1p value to avoid huge loss term.
                                # print("torch.max(dict_q_sample[{}][np_celltype_batch == ct, :]) = {}".format(varname, torch.max(dict_q_sample[varname][np_celltype_batch == ct, :]))): output: ~300 and <800
                                z_incelltype = torch.log(z_incelltype + 1.0)

                            if MODE_COMP_PAIRWISEDIST == 'vanila':
                                pairwise_dist = torch.sum(
                                    (z_incelltype.unsqueeze(0) - z_incelltype.unsqueeze(1)) * (z_incelltype.unsqueeze(0) - z_incelltype.unsqueeze(1)),
                                    2
                                )  # [n, n]
                            else:
                                assert MODE_COMP_PAIRWISEDIST == 'pytorchcdist'
                                pairwise_dist = torch.cdist(
                                    z_incelltype.unsqueeze(0),
                                    z_incelltype.unsqueeze(0),
                                    p=2
                                )[0, :, :]  # [n, n]
                                pairwise_dist = pairwise_dist * pairwise_dist

                            assert (len(pairwise_dist.size()) == 2)


                            loss_zzcloseness += torch.sum(
                                torch.tril(pairwise_dist, diagonal=-1)
                            )/(pairwise_dist.size()[0]*(pairwise_dist.size()[0]-1.0)/2 + 0.0)

                            loss_zzcloseness = loss_zzcloseness / (dict_q_sample[varname].size()[1] + 0.0)  # normalize by dimension as well, because in the case of x_int there might be a log of genes --> the loss becomes huge even after log1p(.)

                    if flag_tensorboardsave:
                        with torch.no_grad():
                            if isinstance(loss_zzcloseness, torch.Tensor):
                                wandb.log(
                                    {"Loss/loss_closeness_{}<-->{} (after mult by coef {})".format(varname, varname, coef_loss_zzcloseness): coef_loss_zzcloseness * loss_zzcloseness},
                                    step=itrcount_wandb
                                )
                            else:
                                wandb.log(
                                    {"Loss/loss_closeness_{}<-->{} (after mult by coef {})".format(varname, varname, coef_loss_zzcloseness): torch.nan},
                                    step=itrcount_wandb
                                )

                    loss = loss + coef_loss_zzcloseness * loss_zzcloseness

            # wandblog other measures ===
            if flag_tensorboardsave:
                with torch.no_grad():
                    wandb.log(
                        {"OtherProbes/pred_x_spl_over_pred_x_int":
                             torch.sum(dict_q_sample['x_spl'])/torch.sum(dict_q_sample['x_int'])
                        },
                        step=itrcount_wandb
                    )


            # update params
            if isinstance(loss, torch.Tensor):  # to handle 1. only imputed loss is active and 2. there is no masking
                loss = loss/(numsteps_accumgrad+0.0)

                if flag_verbose:
                    print(" Before backward() ----- with dl {}".format(idx_current_dl_normal))
                    print("              ---- GPU usage: {}   {}".format(
                        torch.cuda.memory_allocated(),
                        torch.cuda.max_memory_allocated()
                    ))

                loss.backward()
                num_backwards += 1

                if flag_verbose:
                    print(" After backward()----- with dl {}".format(idx_current_dl_normal))
                    print("              ---- GPU usage: {}   {}".format(
                        torch.cuda.memory_allocated(),
                        torch.cuda.max_memory_allocated()
                    ))

                #loss.backward()
                #optim_training.step()
            else:
                assert False

            if (num_backwards > 0) and (num_backwards%numsteps_accumgrad == 0):
                optim_training.step()
                optim_training.zero_grad()
                num_backwards = 0

                if flag_verbose:
                    print("      optim.step() and zero_grad()")



                # update the predictors after GRLs ----
                postGRL_index_dl = -1
                list_iter_dl_afterGRL = [iter(u) for u in list_dl]  # for updating the dual functions separately, to imitate the for loop in the old version.

                #list_iter_dl_afterGRL
                for _ in range(num_updateseparate_afterGRLs):
                    postGRL_index_dl = (postGRL_index_dl + 1)%len(list_iter_dl_afterGRL)

                    optim_training.zero_grad()

                    try:
                        batch_afterGRLs = next(list_iter_dl_afterGRL[postGRL_index_dl])
                    except StopIteration:
                        list_iter_dl_afterGRL[postGRL_index_dl] = iter(list_dl[postGRL_index_dl])
                        batch_afterGRLs = next(list_iter_dl_afterGRL[postGRL_index_dl])


                    # get a 2nd pygdl and q_samples for the below two losses
                    try:
                        batch_2ndpygbatch_postGRL = next(list_iter_dl_2ndpygbatch[(postGRL_index_dl + 1) % len(list_dl)])
                    except StopIteration:
                        list_iter_dl_2ndpygbatch[(postGRL_index_dl + 1) % len(list_dl)] = iter(list_dl[(postGRL_index_dl + 1) % len(list_dl)])
                        batch_2ndpygbatch_postGRL = next(list_iter_dl_2ndpygbatch[(postGRL_index_dl + 1) % len(list_dl)])

                    batch_afterGRLs.INFLOWMETAINF = batch.INFLOWMETAINF
                    batch_2ndpygbatch_postGRL.INFLOWMETAINF = batch.INFLOWMETAINF

                    dict_loss_GRLpreds = self._getloss_GradRevPredictors(
                        batch=batch_afterGRLs,
                        ten_xy_absolute=list_ten_xy_absolute[postGRL_index_dl],
                        ten_xy_touse=list_ten_xy_absolute[postGRL_index_dl],
                        prob_maskknowngenes=prob_maskknowngenes,
                        dict_args_2ndpygbatch={
                            'batch':batch_2ndpygbatch_postGRL,
                            'ten_xy_absolute':list_ten_xy_absolute[(postGRL_index_dl + 1) % len(list_dl)],
                            'ten_xy_touse':list_ten_xy_absolute[(postGRL_index_dl + 1) % len(list_dl)]
                        }
                    )
                    dict_z2notNCC_loss = dict_loss_GRLpreds['z']
                    dict_xbarint2notNCC_loss = dict_loss_GRLpreds['xbarint']
                    dict_xbarint2notbatchID_loss_forGRL = dict_loss_GRLpreds['xbarint2notbatchID']
                    dict_xbarspl2notbatchID_loss_forGRL = dict_loss_GRLpreds['xbarspl2notbatchID']


                    loss_after_GRLs = 0.0
                    for d in [dict_z2notNCC_loss, dict_xbarint2notNCC_loss, dict_xbarint2notbatchID_loss_forGRL, dict_xbarspl2notbatchID_loss_forGRL]:
                        for lossterm_name in d.keys():  # lossterm_name in ['fminf', 'smoothness']
                            loss_after_GRLs = loss_after_GRLs + dict_z2notNCC_loss[lossterm_name]['coef'] * dict_z2notNCC_loss[lossterm_name]['val']

                    loss_after_GRLs.backward()
                    optim_training.step()

                    if flag_verbose:
                        print("              >>>> after GRLs update. (with dl {})".format(postGRL_index_dl))
                        print("              ---- GPU usage: {}   {}".format(
                            torch.cuda.memory_allocated(),
                            torch.cuda.max_memory_allocated()
                        ))

                del list_iter_dl_afterGRL

                optim_training.zero_grad()  # for next accumgrad iters

            itrcount_wandb += 1



        return itrcount_wandb, list_coef_anneal




    def _trainsep_GradRevPreds(
        self,
        optim_gradrevpreds,
        numiters,
        ten_Z,
        ten_CT,
        ten_NCC,
        ten_xbarint,
        ten_xy_absolute,
        device,
        kwargs_dl
    ):


        ds = torch.utils.data.TensorDataset(ten_Z, ten_CT, ten_NCC, ten_xbarint)
        dl = torch.utils.data.DataLoader(ds, shuffle=True, **kwargs_dl)
        iterpygdl_for_afterGRL = iter(dl)

        history_loss = []
        for _ in tqdm(range(numiters)):
            optim_gradrevpreds.zero_grad()

            try:
                batch_afterGRLs = next(iterpygdl_for_afterGRL)
            except StopIteration:
                iterpygdl_for_afterGRL = iter(dl)
                batch_afterGRLs = next(iterpygdl_for_afterGRL)


            y = batch_afterGRLs[2].detach()
            if self.str_modez2notNCCloss_regorclsorwassdist in ['cls', 'wassdist']:
                y = ((y > 0) + 0).float()
            else:
                assert (self.str_modez2notNCCloss_regorclsorwassdist == 'reg')

            dict_z2notNCC_loss = self.crit_loss_z2notNCC(
                z=batch_afterGRLs[0].to(device).detach(),
                module_NCCpredictor=self.module_predictor_z2notNCC,
                ten_CT=batch_afterGRLs[1].detach().to(device),
                ten_NCC=y.detach().to(device)
            ) # NOTE: the first detach is important

            dict_xbarint2notNCC_loss = self.crit_loss_xbarint2notNCC(
                z=batch_afterGRLs[3].to(device).detach(),
                module_NCCpredictor=self.module_predictor_xbarint2notNCC,
                ten_CT=batch_afterGRLs[1].detach().to(device),
                ten_NCC=y.detach().to(device)
            ) # NOTE: the first detach is important

            loss_after_GRLs = 0.0
            for d in [dict_z2notNCC_loss, dict_xbarint2notNCC_loss]:
                for lossterm_name in d.keys():  # lossterm_name in ['fminf', 'smoothness']
                    loss_after_GRLs = loss_after_GRLs + dict_z2notNCC_loss[lossterm_name]['coef'] * dict_z2notNCC_loss[lossterm_name]['val']

            if not isinstance(loss_after_GRLs, torch.Tensor):
                print(">>>>>>>>>> loss_after_GRLs not a torch.Tensor.")
                #raise Exception("dddd")
            else:
                loss_after_GRLs.backward()
                optim_gradrevpreds.step()
                history_loss.append(
                    loss_after_GRLs.detach().cpu().numpy().tolist()
                )

        return history_loss


    def _getloss_GradRevPredictors(self, batch, ten_xy_absolute, ten_xy_touse, prob_maskknowngenes, dict_args_2ndpygbatch):
        '''

        :return: the loss of predictors atop a gradient reverse layer.
        '''

        loss = 0.0
        with torch.no_grad():
            dict_q_sample = self.rsample(
                batch=batch,
                prob_maskknowngenes=prob_maskknowngenes,
                ten_xy_absolute=ten_xy_touse
            )
            dict_q_sample_2ndpygbatch = self.rsample(
                batch=dict_args_2ndpygbatch['batch'],
                prob_maskknowngenes=prob_maskknowngenes,
                ten_xy_absolute=dict_args_2ndpygbatch['ten_xy_touse']
            )

        # Z 2 NotNCC ======================
        if self.coef_z2notNCC_loss > 0.0:
            rng_CT = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
            ]

            rng_NCC = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC']
            ]

            ten_NCC = batch.y[
                :,
                rng_NCC[0]:rng_NCC[1]
            ].to(ten_xy_absolute.device).float()

            if self.str_modez2notNCCloss_regorclsorwassdist in ['cls', 'wassdist']:
                ten_NCC = ((ten_NCC > 0) + 0).float()
            else:
                assert (self.str_modez2notNCCloss_regorclsorwassdist == 'reg')


            dict_z2notNCC_loss = self.crit_loss_z2notNCC(
                z=predadjmat.grad_reverse(
                    dict_q_sample['param_q_cond4flow']['mu_z'].detach()
                ).detach(),
                module_NCCpredictor=self.module_predictor_z2notNCC,
                ten_CT=batch.y[:, rng_CT[0]:rng_CT[1]],
                ten_NCC=ten_NCC.detach()
            )  # NOTE: the first detach is important
            # TODO: should it be on non-cental nodes as well?



        # xbarint 2 NotNCC ================
        if self.coef_xbarint2notNCC_loss > 0.0:
            rng_CT = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT']
            ]

            rng_NCC = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC']
            ]
            ten_NCC = batch.y[
                :batch.batch_size,
                rng_NCC[0]:rng_NCC[1]
            ].to(ten_xy_absolute.device).float()

            if self.str_modexbarint2notNCCloss_regorclsorwassdist in ['cls', 'wassdist']:
                ten_NCC = ((ten_NCC > 0) + 0).float()
            else:
                raise NotImplementedError("ddd")
                assert (self.str_modexbarint2notNCCloss_regorclsorwassdist == 'reg')

            dict_xbarint2notNCC_loss = self.crit_loss_xbarint2notNCC(
                z=predadjmat.grad_reverse(
                    dict_q_sample['param_q_xbarint'][:batch.batch_size].detach()
                ),
                module_NCCpredictor=self.module_predictor_xbarint2notNCC,
                ten_CT=batch.y[:batch.batch_size, rng_CT[0]:rng_CT[1]],
                ten_NCC=ten_NCC.detach()[:batch.batch_size]
            )  # NOTE: the first detach is important
            # TODO: should it be on non-cental nodes as well?


        # add xbarint-->notBatchID loss ===
        if self.coef_xbarint2notbatchID_loss > 0.0:
            rng_batchemb = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb']
            ]

            dict_xbarint2notbatchID_loss = self.crit_loss_xbarint2notbatchID(
                z=predadjmat.grad_reverse(
                    torch.cat(
                        [dict_q_sample['param_q_xbarint'].detach(), dict_q_sample_2ndpygbatch['param_q_xbarint'].detach()],
                        0
                    )
                ),
                module_BatchIDpredictor=self.module_predictor_xbarint2notbatchID,
                ten_BatchID=torch.cat(
                    [batch.y[:, rng_batchemb[0]:rng_batchemb[1]], dict_args_2ndpygbatch['batch'].y[:, rng_batchemb[0]:rng_batchemb[1]]],
                    0
                ).to(ten_xy_absolute.device).detach()
            )  # NOTE: the first detach is important
            # TODO: should it be only on central nodes?



        # add xbarspl-->notBatchID loss ===
        if self.coef_xbarspl2notbatchID_loss > 0.0:
            rng_batchemb = [
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'],
                batch.INFLOWMETAINF['dim_u_int'] + batch.INFLOWMETAINF['dim_u_spl'] + batch.INFLOWMETAINF['dim_CT'] + batch.INFLOWMETAINF['dim_NCC'] + batch.INFLOWMETAINF['dim_BatchEmb']
            ]

            dict_xbarspl2notbatchID_loss = self.crit_loss_xbarspl2notbatchID(
                z=predadjmat.grad_reverse(
                    torch.cat(
                        [dict_q_sample['param_q_xbarspl'].detach(), dict_q_sample_2ndpygbatch['param_q_xbarspl'].detach()],
                        0
                    )
                ),
                module_BatchIDpredictor=self.module_predictor_xbarspl2notbatchID,
                ten_BatchID=torch.cat(
                    [batch.y[:,rng_batchemb[0]:rng_batchemb[1]], dict_args_2ndpygbatch['batch'].y[:,rng_batchemb[0]:rng_batchemb[1]]],
                    0
                ).to(ten_xy_absolute.device).detach()
            )  # NOTE: the first detach is important
            # TODO: should it be only on central nodes?






        # add Z rank loss  ======================================================================================
        if self.coef_rankloss_Z > 0.0:
            raise NotImplementedError(
                "This function is not implemented for `coef_rankloss_Z > 0`"
            )
            assert (batch.n_id.shape[0] == dict_q_sample['xbar_spl'].shape[0])
            assert (ten_xy_absolute.size()[1] == 2)
            ten_x, ten_y = ten_xy_absolute[batch.input_id.tolist(), 0].detach(), ten_xy_absolute[batch.input_id.tolist(), 1].detach()  # [N], [N]

            # subsample the mini-batch to define the rank loss
            rng_N = tuple(range(ten_x.size()[0]))
            list_ij_subsample = random.sample(
                [(i, j) for i in rng_N for j in set(rng_N) - {i}],
                k=min(self.num_subsample_XYrankloss, ten_x.size()[0])
            )
            list_i_subsample = [u[0] for u in list_ij_subsample]
            list_j_subsample = [u[1] for u in list_ij_subsample]

            netout_rank_Xpos = self.module_predictor_ranklossZ_X(
                predadjmat.grad_reverse(
                    torch.cat(
                        [
                            dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_i_subsample, :].detach(),
                            dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_j_subsample, :].detach()
                        ],
                        1
                    )
                )
            )  # [N,2]  # TODO: should it be on non-cental nodes as well?
            assert (netout_rank_Xpos.size()[1] == 2)

            netout_rank_Ypos = self.module_predictor_ranklossZ_Y(
                predadjmat.grad_reverse(
                    torch.cat(
                        [
                            dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_i_subsample, :].detach(),
                            dict_q_sample['param_q_cond4flow']['mu_z'][:batch.batch_size][list_j_subsample, :].detach()
                        ],
                        1
                    )
                )
            )  # [N,2]  # TODO: should it be on non-cental nodes as well?
            # NOTE: the first .detach() is IMPORTANT
            assert (netout_rank_Ypos.size()[1] == 2)

            loss_rank_Xpos = self.crit_Z_rankloss(
                netout_rank_Xpos[:, 0],
                netout_rank_Xpos[:, 1],
                (ten_x[list_i_subsample] - ten_x[list_j_subsample]).sign()
            )
            loss_rank_Ypos = self.crit_Z_rankloss(
                netout_rank_Ypos[:, 0],
                netout_rank_Ypos[:, 1],
                (ten_y[list_i_subsample] - ten_y[list_j_subsample]).sign()
            )
            loss_rank_XYpos_Z = loss_rank_Xpos + loss_rank_Ypos
            loss = loss + loss_rank_XYpos_Z

        # add xbarint rank loss  =====================================================
        if self.coef_rankloss_xbarint:
            raise NotImplementedError(
                "This function is not implemented for `coef_rankloss_xbarint > 0`"
            )
            assert (batch.n_id.shape[0] == dict_q_sample['xbar_spl'].shape[0])
            assert (ten_xy_absolute.size()[1] == 2)
            ten_x, ten_y = ten_xy_absolute[batch.input_id.tolist(), 0].detach(), ten_xy_absolute[batch.input_id.tolist(), 1].detach()  # [N], [N]

            # subsample the mini-batch to define the rank loss
            rng_N = tuple(range(ten_x.size()[0]))
            list_ij_subsample = random.sample(
                [(i, j) for i in rng_N for j in set(rng_N) - {i}],
                k=min(self.num_subsample_XYrankloss, ten_x.size()[0])
            )
            list_i_subsample = [u[0] for u in list_ij_subsample]
            list_j_subsample = [u[1] for u in list_ij_subsample]

            netout_rank_Xpos = self.module_predictor_ranklossxbarint_X(
                predadjmat.grad_reverse(
                    torch.cat(
                        [
                            dict_q_sample['xbar_int'][:batch.batch_size][list_i_subsample, :].detach(),
                            dict_q_sample['xbar_int'][:batch.batch_size][list_j_subsample, :].detach()
                        ],
                        1
                    )
                )
            )  # [N,2]
            assert (netout_rank_Xpos.size()[1] == 2)

            netout_rank_Ypos = self.module_predictor_ranklossxbarint_Y(
                predadjmat.grad_reverse(
                    torch.cat(
                        [
                            dict_q_sample['xbar_int'][:batch.batch_size][list_i_subsample, :].detach(),
                            dict_q_sample['xbar_int'][:batch.batch_size][list_j_subsample, :].detach()
                        ],
                        1
                    )
                )
            )  # [N,2]
            assert (netout_rank_Ypos.size()[1] == 2)

            loss_rank_Xpos = self.crit_xbarint_rankloss(
                netout_rank_Xpos[:, 0],
                netout_rank_Xpos[:, 1],
                (ten_x[list_i_subsample] - ten_x[list_j_subsample]).sign()
            )
            loss_rank_Ypos = self.crit_xbarint_rankloss(
                netout_rank_Ypos[:, 0],
                netout_rank_Ypos[:, 1],
                (ten_y[list_i_subsample] - ten_y[list_j_subsample]).sign()
            )

            loss_rank_XYpos_xbarint = loss_rank_Xpos + loss_rank_Ypos
            loss = loss + loss_rank_XYpos_xbarint

        return {
            'z':dict_z2notNCC_loss,
            'xbarint':dict_xbarint2notNCC_loss,
            'xbarint2notbatchID':dict_xbarint2notbatchID_loss,
            'xbarspl2notbatchID':dict_xbarspl2notbatchID_loss
        }



    def train_imputer(
        self,
        dl: NeighborLoader,
        prob_maskknowngenes: float,
        t_num_steps: int,
        ten_xy_absolute: torch.Tensor,
        optim_training: torch.optim.Optimizer,
        tensorboard_stepsize_save: int,
        itrcount_wandbstep_input: int | None = None,
        numsteps_accumgrad:int=10,
        prob_applytfm_affinexy:float=0.5
    ):
        '''
        One epoch of the training.
        :param dl: pyg's neighborloader.
        :param prob_maskknowngenes: probability of masking some known gene expressions (per-cell)
            to define the self-supervised loss.
        :param t_num_steps: the number of time-steps to be used by the NeuralODE module.
        :param ten_xy_absolute: absoliute xy positions of all cells.
        :param optim_training: the optimizer.
        :param tensorboard_stepsize_save
        :param itrcount_wandbstep_input
        :param numsteps_accumgrad
        :param prob_applytfm_affinexy: the probability that the xy positions go through an affine augmentation.
        :return:
        '''
        raise NotImplementedError(
            "Not implemented."
        )

        # make the affine xy augmenter
        tfm_affinexy = utils_imputer.RandomGeometricTfm(
            prob_applytfm=prob_applytfm_affinexy,
            rng_00=[1.0, 1.0],
            rng_01=[0.0, 0.0],
            rng_10=[0.0, 0.0],
            rng_11=[1.0, 1.0]
        )  # TODO: maybe tune the ranges?

        if itrcount_wandbstep_input is not None:
            itrcount_wandb = itrcount_wandbstep_input + 0
        else:
            itrcount_wandb = 0


        cnt_backward = 0
        optim_training.zero_grad()
        for batch in tqdm(dl):

            ten_xy_touse = tfm_affinexy.forward(ten_xy=ten_xy_absolute)

            flag_tensorboardsave = (itrcount_wandb % tensorboard_stepsize_save == 0)
            loss = 0.0
            dict_q_sample = self.rsample(
                batch=batch,
                prob_maskknowngenes=prob_maskknowngenes,
                ten_xy_absolute=ten_xy_touse
            )
            if dict_q_sample['loss_imputex'] is not None:
                loss = loss + (dict_q_sample['loss_imputex'].mean())/(numsteps_accumgrad+0.0)

            if isinstance(loss, torch.Tensor):
                loss.backward()
                cnt_backward += 1

            if (cnt_backward%numsteps_accumgrad == 0) and (cnt_backward > 0):
                optim_training.step()
                optim_training.zero_grad()




            if flag_tensorboardsave:
                with torch.no_grad():
                    if dict_q_sample['loss_imputex'] is not None:
                        wandb.log(
                            {"Loss/loss_imputex": dict_q_sample['loss_imputex'].mean()},
                            step=itrcount_wandb
                        )
                    else:
                        wandb.log(
                            {"Loss/loss_imputex": torch.nan},
                            step=itrcount_wandb
                        )

            itrcount_wandb += 1

        return itrcount_wandb




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
            'mu_z':{},
            'x_int':{},
            'x_spl':{}
        }  # TODO: add other variables.
        cnt_tqdm = 1
        for batch in tqdm(dl, desc='Epoch {}'.format(cnt_tqdm), position=0, leave=False):

            batch.INFLOWMETAINF = {
                "dim_u_int": self.module_genmodel.dict_varname_to_dim['u_int'],
                "dim_u_spl": self.module_genmodel.dict_varname_to_dim['u_spl'],
                "dim_CT": self.module_genmodel.dict_varname_to_dim['CT'],
                "dim_NCC": self.module_genmodel.dict_varname_to_dim['NCC'],
                "dim_BatchEmb": self.module_genmodel.dict_varname_to_dim['BatchEmb']
            }  # how batch.y is split between u_int, u_spl, CT, and NCC

            cnt_tqdm += 1
            curr_dict_qsample = self.rsample(
                batch=batch,
                prob_maskknowngenes=0.0,
                ten_xy_absolute=ten_xy_absolute
            )
            if isinstance(curr_dict_qsample['ten_out_imputer'], torch.Tensor):
                np_out_imputer = curr_dict_qsample['ten_out_imputer'].detach().cpu().numpy()
            else:
                np_out_imputer = None
            np_muxint = curr_dict_qsample['params_q_impanddisentgl']['muxint'].detach().cpu().numpy()
            np_muxspl = curr_dict_qsample['params_q_impanddisentgl']['muxspl'].detach().cpu().numpy()
            np_muxbar_int = curr_dict_qsample['param_q_xbarint'].detach().cpu().numpy()
            np_muxbar_spl = curr_dict_qsample['param_q_xbarspl'].detach().cpu().numpy()
            np_mu_sin = curr_dict_qsample['param_q_cond4flow']['mu_sin'].detach().cpu().numpy()
            np_mu_sout = curr_dict_qsample['param_q_cond4flow']['mu_sout'].detach().cpu().numpy()
            np_mu_z = curr_dict_qsample['param_q_cond4flow']['mu_z'].detach().cpu().numpy()
            for n_local, n_global in enumerate(batch.input_id.tolist()):
                if np_out_imputer is not None:
                    dict_var_to_dict_nglobal_to_value['output_imputer'][n_global] = np_out_imputer[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxint'][n_global] = np_muxint[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxspl'][n_global] = np_muxspl[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxbar_int'][n_global] = np_muxbar_int[n_local, :]
                dict_var_to_dict_nglobal_to_value['muxbar_spl'][n_global] = np_muxbar_spl[n_local, :]
                dict_var_to_dict_nglobal_to_value['mu_sin'][n_global] = np_mu_sin[n_local, :]
                dict_var_to_dict_nglobal_to_value['mu_sout'][n_global] = np_mu_sout[n_local, :]
                dict_var_to_dict_nglobal_to_value['mu_z'][n_global] = np_mu_z[n_local, :]
                # and generated samples
                dict_var_to_dict_nglobal_to_value['x_int'][n_global] = curr_dict_qsample['x_int'][n_local, :].detach().cpu().numpy()
                dict_var_to_dict_nglobal_to_value['x_spl'][n_global] = curr_dict_qsample['x_spl'][n_local,:].detach().cpu().numpy()

        self.train()

        # create dict_varname_to_output
        dict_varname_to_output = {}
        for k in dict_var_to_dict_nglobal_to_value.keys():
            if(
                set(dict_var_to_dict_nglobal_to_value[k].keys()) != set(range(ten_xy_absolute.size()[0]))
            ):
                if dl.batch_sampler is None:  # i.e., if the cusotimized batch_sampler is not used.
                    assert (k == 'output_imputer')
                    assert (np_out_imputer is None)

            if (k == 'output_imputer') and (np_out_imputer is None):
                dict_varname_to_output[k] = None
            else:

                '''
                print("k={}".format(k))
                print(" ----- list of node indices with no value.")
                for n in range(ten_xy_absolute.size()[0]):
                    if n not in dict_var_to_dict_nglobal_to_value[k].keys():
                        print(" ------------- {}".format(n))
                '''

                dict_varname_to_output[k] = np.stack(
                    [dict_var_to_dict_nglobal_to_value[k][n] for n in range(ten_xy_absolute.size()[0])],
                    0
                )

        return dict_varname_to_output



    def _check_args(self):

        # related to xbarint2notbatchID and xbarsplt2notbatchID
        assert isinstance(self.module_predictor_xbarint2notbatchID, predictorbatchID.PredictorBatchID)
        assert isinstance(self.coef_xbarint2notbatchID_loss, float)
        assert isinstance(self.module_predictor_xbarspl2notbatchID, predictorbatchID.PredictorBatchID)
        assert isinstance(self.coef_xbarspl2notbatchID_loss, float)


        assert isinstance(self.module_varphi_enc_int, varphienc4xbar.EncX2Xbar)
        assert isinstance(self.module_varphi_enc_spl, varphienc4xbar.EncX2Xbar)

        assert isinstance(self.coef_rankloss_Z, float)
        assert(self.coef_rankloss_Z >= 0.0)
        assert isinstance(self.module_predictor_ranklossZ_X, nn.Module)
        assert isinstance(self.module_predictor_ranklossZ_Y, nn.Module)


        assert (isinstance(self.coef_z2notNCC_loss, float))
        assert (self.coef_z2notNCC_loss >= 0.0)
        assert isinstance(self.module_predictor_z2notNCC, nn.Module)
        assert (self.str_modez2notNCCloss_regorclsorwassdist in ['reg', 'cls', 'wassdist'])

        assert isinstance(self.coef_xbarint2notNCC_loss, float)
        assert (self.coef_xbarint2notNCC_loss >= 0.0)
        assert isinstance(self.module_predictor_xbarint2notNCC, nn.Module)
        assert isinstance(self.str_modexbarint2notNCCloss_regorclsorwassdist, str)
        assert (self.str_modexbarint2notNCCloss_regorclsorwassdist in ['reg', 'cls', 'wassdist'])

        assert isinstance(self.num_subsample_XYrankloss, int)
        assert (self.num_subsample_XYrankloss > 0)
        assert isinstance(self.coef_rankloss_xbarint, float)
        assert (self.coef_rankloss_xbarint >= 0.0)
        assert isinstance(self.module_predictor_ranklossxbarint_X, nn.Module)
        assert isinstance(self.module_predictor_ranklossxbarint_Y, nn.Module)

        assert (
            isinstance(self.coef_xbarintCT_loss, float)
        )
        assert (self.coef_xbarintCT_loss >= 0.0)
        assert(
            isinstance(self.module_classifier_xbarintCT, nn.Module)
        )
        assert (
            isinstance(self.coef_xbarsplNCC_loss, float)
        )
        assert (self.coef_xbarsplNCC_loss >= 0.0)
        assert (
            isinstance(self.module_predictor_xbarsplNCC, nn.Module)
        )
        assert (
            isinstance(self.str_modexbarsplNCCloss_regorcls, str)
        )
        assert (
            self.str_modexbarsplNCCloss_regorcls in ['reg', 'cls']
        )



        if self.flag_drop_loss_logQdisentangler:
            # in this case moduleDisent.std is not trained --> must be set to a fixed number.
            self.module_impanddisentgl : GNNDisentangler
            assert (
                self.module_impanddisentgl.std_minval_finalclip == self.module_impanddisentgl.std_maxval_finalclip
            )
        if self.weight_logprob_zinbpos != -1:
            assert (self.weight_logprob_zinbpos >= 0.0)
        else:
            assert (self.weight_logprob_zinbzero == -1)

        if self.weight_logprob_zinbzero != -1:
            assert (self.weight_logprob_zinbzero >= 0.0)
        else:
            assert (self.weight_logprob_zinbpos == -1)



        assert (
            self.str_modeP3loss_regorcls in ['reg', 'cls']
        )

        assert (
            isinstance(self.module_cond4flowvarphi0, Cond4FlowVarphi0) or isinstance(self.module_cond4flowvarphi0, Cond4FlowVarphi0SimpleMLPs)
        )

        assert (
            isinstance(self.module_conditionalflowmatcher, utils_flowmatching.ConditionalFlowMatcher)
        )

        # check if the passed flag_use_int_u and flag_use_spl_u are consistent in vardist and cond4flow
        assert (
            self.module_cond4flowvarphi0.kwargs_genmodel['flag_use_int_u'] == self.module_genmodel.flag_use_int_u
        )
        assert (
            self.module_cond4flowvarphi0.kwargs_genmodel['flag_use_spl_u'] == self.module_genmodel.flag_use_spl_u
        )
        assert (
            self.module_cond4flowvarphi0.kwargs_genmodel['flag_use_int_u'] in [True, False]
        )
        assert (
            self.module_cond4flowvarphi0.kwargs_genmodel['flag_use_spl_u'] in [True, False]
        )


        # check if the passed flag_use_int_u and flag_use_spl_u are consistent in vardist and disentangler
        assert (
            self.module_genmodel.flag_use_int_u == self.module_impanddisentgl.flag_use_int_u
        )
        assert (
            self.module_genmodel.flag_use_spl_u == self.module_impanddisentgl.flag_use_spl_u
        )
        assert (
            self.module_genmodel.flag_use_int_u in [True, False]
        )
        assert (
            self.module_genmodel.flag_use_spl_u in [True, False]
        )
        assert (
            self.module_impanddisentgl.flag_use_int_u in [True, False]
        )
        assert (
            self.module_impanddisentgl.flag_use_spl_u in [True, False]
        )



        if isinstance(self.module_impanddisentgl, Disentangler):
            raise NotImplementedError("ddd")
            self.module_impanddisentgl:Disentangler
            if self.module_impanddisentgl.str_mode_headxint_headxspl_headboth in ['headboth']:  # for the Disentangler module
                if self.module_genmodel.dict_pname_to_scaleandunweighted['x'] == [None, None]:
                    raise Exception(
                        "Disentangler.str_mode_headxint_headxspl_headboth is set to 'headboth' and dict_pname_to_scaleandunweighted['x'] is set to [None, None]." +\
                        "Doing this is problematic because nothing ensures that x = x_int + x_spl."
                    )
            else:
                assert (
                    self.module_impanddisentgl.str_mode_headxint_headxspl_headboth in ['headxint', 'headxspl']
                )
                if self.module_genmodel.dict_pname_to_scaleandunweighted['x'] != [None, None]:
                    raise Exception(
                        "Disentangler.str_mode_headxint_headxspl_headboth is set to {} and dict_pname_to_scaleandunweighted['x'] is not set to [None, None]." +\
                        "Doing this is problematic because x = x_int + x_spl is enforced by design, so it shouldn't be enforced by p(x | x_int, x_spl).".format(
                            self.module_impanddisentgl.str_mode_headxint_headxspl_headboth
                        )
                    )

        if isinstance(self.module_impanddisentgl, GNNDisentangler):
            self.module_impanddisentgl:GNNDisentangler
            if self.module_impanddisentgl.str_mode_headxint_headxspl_headboth in ['headboth', 'twosep']:  # for the Disentangler module
                if self.module_genmodel.dict_pname_to_scaleandunweighted['x'] == [None, None]:
                    raise Exception(
                        "Disentangler.str_mode_headxint_headxspl_headboth is set to '...' and dict_pname_to_scaleandunweighted['x'] is set to [None, None]." +\
                        "Doing this is problematic because nothing ensures that x = x_int + x_spl."
                    )
            else:
                assert (
                    self.module_impanddisentgl.str_mode_headxint_headxspl_headboth in ['headxint', 'headxspl']
                )
                if self.module_genmodel.dict_pname_to_scaleandunweighted['x'] != [None, None]:
                    raise Exception(
                        "Disentangler.str_mode_headxint_headxspl_headboth is set to {} and dict_pname_to_scaleandunweighted['x'] is not set to [None, None]." +\
                        "Doing this is problematic because x = x_int + x_spl is enforced by design, so it shouldn't be enforced by p(x | x_int, x_spl).".format(
                            self.module_impanddisentgl.str_mode_headxint_headxspl_headboth
                        )
                    )

        if isinstance(self.module_impanddisentgl, DisentanglerTwoSep):
            raise NotImplementedError("ddd")
            if self.module_genmodel.dict_pname_to_scaleandunweighted['x'] == [None, None]:
                raise Exception(
                    "DisentanglerTwoSep is used along with dict_pname_to_scaleandunweighted['x'] set to [None, None]." + \
                    "Doing this is problematic because nothing ensures that x = x_int + x_spl."
                )

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
        assert(
            isinstance(self.module_impanddisentgl, Disentangler) or isinstance(self.module_impanddisentgl, DisentanglerTwoSep) or isinstance(self.module_impanddisentgl, GNNDisentangler)
        )









