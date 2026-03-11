
"""
Utilities for providing guidance (or steering) for insilico generation.
"""

from abc import ABC, abstractmethod
from scipy import stats
import numpy as np
import pandas as pd
import torch

class GenerationGuider(ABC):
    """
    General guider for the in-silico generation phase of mintflow.
    """

    @abstractmethod
    def get_guidanceloss_and_trackinginfo(self, x:torch.Tensor):
        pass

    @abstractmethod
    def modify_Sout_Z(self, z:torch.Tensor, s_out:torch.Tensor, loc_z:torch.Tensor, sigma_z:torch.Tensor, loc_sout:torch.Tensor, sigma_sout:torch.Tensor):
        pass


class ConfIntervalProjectorGuider(GenerationGuider):
    """
    A guider that simply projects z and s_out to the confidence interval of priors, so the cell type labels provided for in-silico genration are not ignored.
    """

    def __init__(self, conf_interval:float, *args, **kwargs):
        self.conf_interval = conf_interval
        super().__init__(*args, **kwargs)
    

    @torch.no_grad()
    def _project_X(self, ten_X:torch.Tensor, ten_loc:torch.Tensor, ten_sigma:torch.Tensor, conf_interval:float):
        """
        Given a set of points in `ten_X` and the corresponding normal distribution from which they are sampled (specified in `ten_sigma` and `ten_loc`),
        this function projects each point to the `conf_interval` confidence interval of its distribution.
        
        :param ten_X: Description
        :type ten_X: torch.Tensor
        :param ten_loc: Description
        :type ten_loc: torch.Tensor
        :param ten_sigma: Description
        :type ten_sigma: torch.Tensor
        :param conf_interval: Description
        :type conf_interval: float
        """
        # arg checks ===
        assert isinstance(ten_X, torch.Tensor)
        assert isinstance(ten_loc, torch.Tensor)
        assert isinstance(ten_sigma, torch.Tensor)
        assert isinstance(conf_interval, float)
        assert conf_interval >= 0.0
        assert conf_interval < 1.0
        assert len(ten_X.size()) == 2
        assert len(ten_loc.size()) == 2
        assert len(ten_sigma.size()) == 2

        # compute the D2 (i.e. Mahalanobis distane per cell)
        D2 = (ten_X - ten_loc) * (ten_X - ten_loc)  # [num_cell x D]
        D2 = D2 / (ten_sigma * ten_sigma)  # [num_cell x D]
        D2 = torch.sum(D2, 1)  # [num_cell]


        # find out for which cells the vector has to be projected
        val_chi2 = stats.chi2.ppf(
            q=conf_interval,
            df=ten_X.size()[1]
        )  # float
        ten_flag_do_project = ((D2 > val_chi2) + 0).unsqueeze(1)  # [num_cell x 1]

        # compute the projection 
        ten_projected_X = ten_loc + (ten_X - ten_loc) * (torch.sqrt(val_chi2 / D2).unsqueeze(1))  # [num_cell x D]

        # create the output based on `ten_projected_X` and `ten_flag_do_project`
        return (ten_projected_X * ten_flag_do_project) + (ten_X * (1.0 - ten_flag_do_project)), ten_flag_do_project
        

    def modify_Sout_Z(self, z:torch.Tensor, s_out:torch.Tensor, loc_z:torch.Tensor, sigma_z:torch.Tensor, loc_sout:torch.Tensor, sigma_sout:torch.Tensor):

        with torch.no_grad():
            # args checks ===
            assert isinstance(z, torch.Tensor)  # [num_cells x D]
            assert isinstance(loc_z, torch.Tensor)  # [num_cells x D]
            assert isinstance(sigma_z, torch.Tensor)  # [num_cells x D]

            assert isinstance(s_out, torch.Tensor)  # [num_cells x D]
            assert isinstance(loc_sout, torch.Tensor)  # [num_cells x D]
            assert isinstance(sigma_sout, torch.Tensor)  # [num_cells x D]
            
            # get `new_z` and `new_sout`
            new_z, z_flag_do_project = self._project_X(
                ten_X=z,
                ten_loc=loc_z,
                ten_sigma=sigma_z,
                conf_interval=self.conf_interval
            )
            new_sout, sout_flag_do_project = self._project_X(
                ten_X=s_out,
                ten_loc=loc_sout,
                ten_sigma=sigma_sout,
                conf_interval=self.conf_interval
            )

            # modify `z` and `s_out` in place
            z.copy_(new_z)
            s_out.copy_(new_sout)

            return dict(
                z_flag_do_project=z_flag_do_project.detach().cpu().numpy().flatten(),
                sout_flag_do_project=sout_flag_do_project.detach().cpu().numpy().flatten()
            )  # to be used for visualization
            

        
    
class GenePerturbationGuider(ConfIntervalProjectorGuider):

    def __init__(self, np_desired_expression:np.ndarray, np_mask:np.ndarray, device, *args, **kwargs):

        # check args ===
        assert isinstance(np_desired_expression, np.ndarray)
        assert isinstance(np_mask, np.ndarray)
        assert np_mask.dtype == bool
        assert np_mask.shape[0] == np_desired_expression.shape[0]
        assert np_mask.shape[1] == np_desired_expression.shape[1]

        # grab args ==
        self.ten_desired_expression = torch.tensor(np_desired_expression, device=device, requires_grad=False)
        self.ten_mask = torch.tensor(np_mask, device=device, requires_grad=False)

        # call on super
        super().__init__(*args, **kwargs)

    def get_guidanceloss_and_trackinginfo(self, x:torch.Tensor):
        # check args
        assert len(x.size()) == 2
        assert x.size()[0] == self.ten_desired_expression.size()[0]
        assert x.size()[1] == self.ten_desired_expression.size()[1]

        # compute the loss to be optimised
        ten_loss = (x - self.ten_desired_expression) * (x - self.ten_desired_expression)  # [num_cell x num_gene]
        loss_per_elem = torch.masked_select(
            ten_loss,
            self.ten_mask
        )  # [num_nonzero]
        loss_4backward = torch.mean(loss_per_elem)

        # compute the tracking info
        dict_rowcol_to_lossval = dict()
        with torch.no_grad():
            np_indices = torch.nonzero(self.ten_mask).detach().cpu().numpy()  # [num_nonzero x 2], for [row_index, col_index]
            for idx_elem in range(np_indices.shape[0]):
                dict_rowcol_to_lossval['cellindex_{}_geneindex_{}'.format(
                    np_indices[idx_elem, 0],
                    np_indices[idx_elem, 1]
                )] = loss_per_elem[idx_elem].detach().cpu().numpy().tolist()
        
        # return 
        return loss_4backward, dict_rowcol_to_lossval



