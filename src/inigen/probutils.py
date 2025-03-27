
import numpy as np
import torch
from torch.distributions.normal import Normal


class ExtenededNormal:
    '''
    A wrapper for Normal that allws for
        - unweigted loglik
        - scale=0.0, i.e. deterministic output.
    '''
    def __init__(self, loc, scale, flag_unweighted):
        self.loc = loc
        self.scale = scale
        self.flag_unweighted = flag_unweighted
        # assert(
        #     isinstance(self.scale, float) or isinstance(self.scale, torch.Tensor)
        # )
        # TODO: trainable scale parameter is not implemented yet.

    def rsample(self, num_samples=1):
        if self.scale > 0.0:
            if num_samples > 1:
                return Normal(
                    loc=self.loc,
                    scale=self.scale
                ).rsample(torch.Size([num_samples]))
            else:
                return Normal(
                    loc=self.loc,
                    scale=self.scale
                ).rsample()
        else:
            assert (self.scale == 0.0)
            if num_samples > 1:
                return torch.stack(num_samples*[self.loc+0.0], 0)
            else:
                return self.loc+0.0

    def sample(self, num_samples=1):
        with torch.no_grad():
            return self.rsample(num_samples)

    def log_prob(self, ten_in):
        # determine number of samples
        if ten_in.size() == self.loc.size():
            num_samples = None
        else:
            # print("ten_in.size() = {}".format(ten_in.size()))
            # print("self.loc.size() = {}".format(self.loc.size()))
            assert(
                len(ten_in.size()) == (len(self.loc.size())+1)
            )
            assert(
                list(ten_in.size()[1::]) == list(self.loc.size())
            )
            num_samples = ten_in.size()[0]

        if self.scale > 0.0:
            if self.flag_unweighted:
                if num_samples is not None:  # if ten_in contains more than one sample --> unsqueeze(0).
                    return -(ten_in-self.loc.unsqueeze(0))**2
                else:
                    return -(ten_in - self.loc) ** 2
            else:
                return Normal(
                    loc=self.loc,
                    scale=self.scale
                ).log_prob(ten_in)
        else:
            assert(self.scale == 0.0)
            if not self.flag_unweighted:
                assert Exception(
                    "scale and flag_unweighted are set to {} and {} ---> loglik becomes infinity.".format(
                        self.scale,
                        self.flag_unweighted
                    )
                )
                # return torch.inf  #scale=0 and flag_unweight=False means it's deterministic.
            # Note: the below term is necessary, because the samples that go through
            # the loglikP(.) are not generated from P(.) itself.
            return -(ten_in-self.loc.unsqueeze(0))**2







