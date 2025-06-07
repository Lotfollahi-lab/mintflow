'''
Different samplers for z and s vectors.
- from N(0,1)
- a fixed random vector per cell-type.
- ...
'''

from typing import List
import torch
from torch.distributions.normal import Normal
from abc import ABC, abstractmethod


class ZSSampler(ABC):
    def __init__(self, list_celltype):
        self.list_celltype = list_celltype

    @abstractmethod
    def gen_zs(self, N:int, D:int):
        '''
        :param N
        :param D
        :return: ten_output: a tensor of shape [N, D].
        '''
        pass

class RandomZSSampler(ZSSampler):
    '''
    Ignores the cell-types and generates from N(0,1).
    '''
    def gen_zs(self, N:int, D:int):
        return Normal(
            loc=torch.zeros([N, D]),
            scale=torch.tensor([1.0])
        ).sample()  # [N, D]


class PerCelltypeZSSampler(ZSSampler):
    def gen_zs(self, N:int, D:int):
        list_celltype = self.list_celltype
        assert (isinstance(list_celltype, List))
        assert (len(list_celltype) == N)
        assert (
            set(list_celltype) == set(range( max(list_celltype) + 1 ))
        )
        num_celltypes = max(list_celltype) + 1
        x_bank = [
            Normal(
                loc=torch.zeros([D]),
                scale=torch.tensor([1.0])
            ).sample()
            for _ in range(num_celltypes)
        ]

        output = torch.stack(
            [x_bank[list_celltype[n]] for n in range(N)],
            dim=0
        )  # [N, D]

        return output




