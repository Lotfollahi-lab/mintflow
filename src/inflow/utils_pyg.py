
import random
import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from tqdm.notebook import tqdm, trange
from datetime import datetime

from random import shuffle

class PygSTDataGridBatchSampler(Sampler):
    '''
    Creates a grid (with random offset) given the xy positions.
    '''
    def __init__(self, ten_xy :torch.Tensor, width_window :int, min_numpoints_ingrid :int, flag_disable_randoffset:bool):
        '''
        Inputs args.
        :param ten_xy:
        :param width_window: the window size of the grid.
        :param min_numpoints_ingrid: if a grid contains less than `min_numpoints_ingrid` points, no batch will be created.
        :param flag_disable_randoffset: if set to True, no random offset is applied. To be used for, e.g., evaluation phase.
        '''
        self.width_window = width_window
        self.ten_xy = ten_xy.detach().clone()
        self.min_numpoints_ingrid = min_numpoints_ingrid
        self.flag_disable_randoffset = flag_disable_randoffset

        # the grid gets a random offset --> __len__ changes --> a typical len is takes and is kept fixed ===
        self.typical_num_batches = np.max(
            [len(self._get_batchlist()) for _ in range(3)]
        )


    def __len__(self):
        return self.typical_num_batches

    def __iter__(self):
        random.seed(datetime.now().timestamp())
        list_batch = self._get_batchlist()

        # adjust lenth of list_batch
        if len(list_batch) > self.typical_num_batches:
            while(len(list_batch) > self.typical_num_batches):
                del list_batch[random.choice(range(len(list_batch)))]
        elif len(list_batch) < self.typical_num_batches:
            while(len(list_batch) < self.typical_num_batches):
                list_batch = list_batch + [list_batch[random.choice(range(len(list_batch)))]]
        assert (len(list_batch) == self.typical_num_batches)

        for batch in list_batch:
            yield batch

    @torch.no_grad()
    def _get_batchlist(self, eps_dilatewin=10.0):
        '''
        Places a grid on the xy position with a random offset.
        If the window is non-empty, it creates a batch for it.
        '''
        if not self.flag_disable_randoffset:
            offset_x, offset_y = int(np.random.rand() * self.width_window), int(np.random.rand() * self.width_window)
        else:
            offset_x, offset_y = 0, 0

        list_x = list(
            np.arange(
                self.ten_xy[:,0].min().detach().cpu().numpy().tolist() - offset_x,
                self.ten_xy[:,0].max().detach().cpu().numpy().tolist(),
                self.width_window
            )
        )
        list_y = list(
            np.arange(
                self.ten_xy[:,1].min().detach().cpu().numpy().tolist() - offset_y,
                self.ten_xy[:,1].max().detach().cpu().numpy().tolist(),
                self.width_window
            )
        )
        list_batch = []
        for x in list_x:
            for y in list_y:
                # compute numpoints within the window
                x_begin, x_end = x, x + self.width_window
                y_begin, y_end = y, y + self.width_window
                x_begin, x_end = x_begin - eps_dilatewin, x_end + eps_dilatewin
                y_begin, y_end = y_begin - eps_dilatewin, y_end + eps_dilatewin
                filter_x = ((self.ten_xy[:, 0] >= x_begin) * (self.ten_xy[:, 0] <= x_end))
                filter_y = ((self.ten_xy[:, 1] >= y_begin) * (self.ten_xy[:, 1] <= y_end))
                filter_xy = filter_x * filter_y
                numpoints = torch.sum(
                    filter_xy
                ).detach().cpu().numpy().tolist()
                if numpoints >= self.min_numpoints_ingrid:  # enough points within the grid ==> create a batch
                    list_batch.append(
                        torch.where(filter_xy)[0].detach().cpu().numpy().tolist()
                    )

        # randshufflelist_batch
        shuffle(list_batch)

        return list_batch

    @torch.no_grad()
    def get_maxnumpoints_insquare(self, eps_dilatewin=10.0):
        '''
        Computes the maximal number of cells within any window.
        The idea is that it puts the corner of a window on each cell position and reports the maximum number.
        '''
        max_numpoints = -np.inf
        for n in tqdm(range(self.ten_xy.size()[0])):
            x, y = self.ten_xy[n, 0].detach().cpu().numpy().tolist(), self.ten_xy[
                n, 1].detach().cpu().numpy().tolist()

            # put top-left corner on xy
            x_begin, x_end = x, x + self.width_window
            y_begin, y_end = y, y + self.width_window
            x_begin, x_end = x_begin - eps_dilatewin, x_end + eps_dilatewin
            y_begin, y_end = y_begin - eps_dilatewin, y_end + eps_dilatewin
            filter_x = ((self.ten_xy[:, 0] >= x_begin) * (self.ten_xy[:, 0] <= x_end))
            filter_y = ((self.ten_xy[:, 1] >= y_begin) * (self.ten_xy[:, 1] <= y_end))
            numpoints = torch.sum(
                filter_x * filter_y
            ).detach().cpu().numpy().tolist()
            if numpoints > max_numpoints:
                max_numpoints = numpoints + 0.0

            # put top-right corner on xy
            x_begin, x_end = x - self.width_window, x
            y_begin, y_end = y, y + self.width_window
            x_begin, x_end = x_begin - eps_dilatewin, x_end + eps_dilatewin
            y_begin, y_end = y_begin - eps_dilatewin, y_end + eps_dilatewin
            filter_x = ((self.ten_xy[:, 0] >= x_begin) * (self.ten_xy[:, 0] <= x_end))
            filter_y = ((self.ten_xy[:, 1] >= y_begin) * (self.ten_xy[:, 1] <= y_end))
            numpoints = torch.sum(
                filter_x * filter_y
            ).detach().cpu().numpy().tolist()
            if numpoints > max_numpoints:
                max_numpoints = numpoints + 0.0

            # put bottom-left corner on xy
            x_begin, x_end = x, x + self.width_window
            y_begin, y_end = y - self.width_window, y
            x_begin, x_end = x_begin - eps_dilatewin, x_end + eps_dilatewin
            y_begin, y_end = y_begin - eps_dilatewin, y_end + eps_dilatewin
            filter_x = ((self.ten_xy[:, 0] >= x_begin) * (self.ten_xy[:, 0] <= x_end))
            filter_y = ((self.ten_xy[:, 1] >= y_begin) * (self.ten_xy[:, 1] <= y_end))
            numpoints = torch.sum(
                filter_x * filter_y
            ).detach().cpu().numpy().tolist()
            if numpoints > max_numpoints:
                max_numpoints = numpoints + 0.0

            # put bottom-right corner on xy
            x_begin, x_end = x - self.width_window, x
            y_begin, y_end = y - self.width_window, y
            x_begin, x_end = x_begin - eps_dilatewin, x_end + eps_dilatewin
            y_begin, y_end = y_begin - eps_dilatewin, y_end + eps_dilatewin
            filter_x = ((self.ten_xy[:, 0] >= x_begin) * (self.ten_xy[:, 0] <= x_end))
            filter_y = ((self.ten_xy[:, 1] >= y_begin) * (self.ten_xy[:, 1] <= y_end))
            numpoints = torch.sum(
                filter_x * filter_y
            ).detach().cpu().numpy().tolist()
            if numpoints > max_numpoints:
                max_numpoints = numpoints + 0.0

        return max_numpoints
