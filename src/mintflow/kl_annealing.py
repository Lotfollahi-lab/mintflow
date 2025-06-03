
'''
The implementation of annealing for the KL-terms over z and sout.
'''

from abc import ABC, abstractmethod
import numpy as np

class AnnealingSchedule(ABC):

    def __init__(self, coef_min:float, coef_max:float, num_cyles:int, numepochs_in_cycle:int):
        '''

        :param num_cyles: pass np.inf if  the cycles repeat for ever.
        :param numepochs_in_cycle: number of epochs in each cycle.
        '''
        self.coef_min = coef_min
        self.coef_max = coef_max
        self.num_cyles = num_cyles
        self.numepochs_in_cycle = numepochs_in_cycle

    def __iter__(self):
        coef01_one_cycle = self.func_t2coef_onecycle(
            t_01=(np.array(range(self.numepochs_in_cycle))+0.0) / (self.numepochs_in_cycle-1.0)
        )
        assert (np.min(coef01_one_cycle) >= 0.0)  # because the values are assumed to be in [0,1]
        assert (np.min(coef01_one_cycle) <= 1.0)  # because the values are assumed to be in [0,1]

        coef_minmax_onecycle = (coef01_one_cycle * (self.coef_max - self.coef_min + 0.0)) + self.coef_min

        # the cycles
        if self.num_cyles == np.inf:
            while True:
                for n in range(self.numepochs_in_cycle):
                    yield coef_minmax_onecycle[n]
        else:
            for _ in range(self.num_cyles):
                for n in range(self.numepochs_in_cycle):
                    yield coef_minmax_onecycle[n]


        # coefs kept at coef_max
        while True:
            yield self.coef_max


    @abstractmethod
    def func_t2coef_onecycle(self, t_01:np.ndarray):
        '''
        Given a ndarray of time points in one cycle, this function is expected to return the weights.
        BOTH input and output values are assumed to be in ragne 0 and 1.
        :param t_01:
        :return:
        '''
        pass



class LinearAnnealingSchedule(AnnealingSchedule):
    def func_t2coef_onecycle(self, t_01: np.ndarray):
        return t_01 + 0.0










