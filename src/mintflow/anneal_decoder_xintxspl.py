
'''
The implementation of annealing for decoding loss of xint and xspl.
'''

from abc import ABC, abstractmethod
import numpy as np

class AnnealingDecoderXintXspl:
    def __init__(self, coef_min:float, coef_max:float, num_phase1:int, num_phase2:int):
        """
        Has three phases
        - In phase 1 the coefficients are `coef_min` during `num_phase1` iterations.
        - In phase 2, the coefficient are linearly incerrased to `coef_max` in `num_phase2` iterations.
        :param coef_min:
        :param coef_max:
        :param num_phase1:
        :param num_phase2:
        """
        self.coef_min = coef_min
        self.coef_max = coef_max
        self.num_phase1 = num_phase1
        self.num_phase2 = num_phase2

        if coef_max < coef_min:
            raise Exception(
                "In the annealing module for decoder_XintXspl, coef_max < coef_min."
            )

        if (coef_min < 0.0) or (coef_max < 0.0):
            raise Exception(
                "In the annealing module for decoder_XintXspl coef_min and/or coef_max are set to negative numbers."
            )

    def __iter__(self):
        # phase 1
        for _ in range(self.num_phase1):
            yield self.coef_min

        # phase 2
        for t in range(self.num_phase2):
            if self.num_phase2 == 1:  # to handle divbyzero if `self.num_phase2` equals 1.
                coef_t = 0.5  # ((t + 0.0) / (self.num_phase2 - 1.0))
            else:
                coef_t = ((t + 0.0) / (self.num_phase2 - 1.0))

            yield (self.coef_min + (self.coef_max - self.coef_min) * coef_t)

        # phase 3
        while True:
            yield self.coef_max

