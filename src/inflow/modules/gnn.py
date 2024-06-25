
import copy
import torch
import torch.nn as nn
import torch_geometric as pyg
import gc


def graph_neigh_avg_pool(x:torch.Tensor, edge_index:torch.Tensor, flag_make_checks=False):
    #pyg's neibor avg adds self-loops, but this function avoids that.
    if flag_make_checks:
        assert(
            not pyg.utils.contains_self_loops(edge_index)
        )
        assert(
            torch.all(
                torch.eq(
                    torch.tensor(edge_index),
                    pyg.utils.to_undirected(edge_index)
                )
            )
        )
    row, col = edge_index
    return pyg.utils.scatter(
        x[row], col, dim=0,
        dim_size=1+int(edge_index.max()),
        reduce='mean'
    )

class GrModuleWithLayeredEval(nn.Module):
    '''
    A module that supporst graph layered evaluation approach.
    Assumes that
        - module is entirely in `self.list_modules`
        - each module in the list take in `x` and `edge_index`
    '''
    def __init__(self, dim_input, dim_output):
        super(GrModuleWithLayeredEval, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output

    def forward(self, x, edge_index):
        output = x+0.0
        for mod in self.list_modules:
            output = mod(output, edge_index)
        return output

    @torch.no_grad()
    def evaluate_layered(self, x, edge_index, kwargs_dl):
        '''

        :param x:
        :param edge_index:
        :param kwargs_dl: Additional kwargs of the neighbourloader, like
            - batch_size
            - num_workers
            - ...
        :return:
        '''
        #passed print("Reached here 0"); assert False
        dl = pyg.loader.NeighborLoader(
            pyg.data.Data(x=(x+0.0).to('cpu'), edge_index=copy.copy(edge_index).to("cpu")),
            **{**{'shuffle':False}, **kwargs_dl}
        )

        # ----> NOT PASSEdprint("Reached here xxx"); assert False
        output = x+0.0 #the final output (updated layer by layer)
        for mod in self.list_modules:
            tmp_dict_n_to_netout = {}
            for batch in dl:
                netout = mod(
                    output[batch.n_id.tolist()].to(x.device),
                    batch.edge_index.to(x.device)
                )[:batch.batch_size] #[batch_size, D]
                for n_local, n_global in enumerate(batch.input_id.tolist()):
                    tmp_dict_n_to_netout[n_global] = netout[n_local]
            assert(
                set(tmp_dict_n_to_netout.keys()) == set(range(x.size()[0]))
            )
            output = torch.stack(
                [tmp_dict_n_to_netout[n] for n in range(x.size()[0])],
                0
            ) #[N, D]
        return output

    @torch.no_grad()
    def evaluate_layered_Xsparse(self, x, edge_index, kwargs_dl, device):
        '''
        Feeds a sparse matrix x to a GNN by the layered approach.
            The assumption is that after the first hop, the embeddings h are dense but of lower dimension than
            number of genes. So after the first hope, h is not sparse.
            #TODO: this doesn't happen for inflow. Because x_int and x_spl are of the same shape, and not sparse.
        :param x:
        :param edge_index:
        :param kwargs_dl: Additional kwargs of the neighbourloader, like
            - batch_size
            - num_workers
            - ...
        :return:
        '''
        assert(
            not isinstance(x, torch.Tensor)
        )
        output = torch.sparse_coo_tensor(
            indices=x.nonzero(),
            values=x.data,
            size=x.shape
        )  # the final output (updated layer by layer)
        for idx_mod, mod in enumerate(self.list_modules):
            #create the neighborloader
            dl = pyg.loader.NeighborLoader(
                pyg.data.Data(x=output, edge_index=copy.copy(edge_index).to("cpu")),
                **{**{'shuffle': False}, **kwargs_dl}
            )
            #feed to the current layer
            tmp_dict_n_to_netout = {}
            for batch in dl:
                if idx_mod==0: #if x_sparse is being fed.
                    netout = mod(
                        torch.tensor(batch.x.toarray()).to(device),
                        batch.edge_index.to(device)
                    )[:batch.batch_size].cpu()  # [batch_size, D]
                else: #otherwise
                    netout = mod(
                        batch.x.to(device),
                        batch.edge_index.to(device)
                    )[:batch.batch_size].cpu()  # [batch_size, D]
                for n_local, n_global in enumerate(batch.input_id.tolist()):
                    tmp_dict_n_to_netout[n_global] = netout[n_local]
            assert (
                set(tmp_dict_n_to_netout.keys()) == set(range(x.shape[0]))
            )
            output = torch.stack(
                [tmp_dict_n_to_netout[n] for n in range(x.shape[0])],
                0
            ).cpu()  # [N, D]
            del dl
            gc.collect()
        return output



class AvgpoolLayerWithoutselfloop(nn.Module):
    def forward(self, x, edge_index):
        return graph_neigh_avg_pool(x=x, edge_index=edge_index, flag_make_checks=False)




class KhopAvgPoolWithoutselfloop(GrModuleWithLayeredEval):
    def __init__(self, num_hops, *args, **kwargs):
        super(KhopAvgPoolWithoutselfloop, self).__init__(*args, **kwargs)
        self.num_hops = num_hops
        #create the list of modules ===
        self.list_modules = nn.ModuleList(
            [AvgpoolLayerWithoutselfloop() for _ in range(self.num_hops)]
        )

class SageConvAndActivation(nn.Module):
    '''
    A building block for `SAGE`
    '''
    def __init__(self, module_sageconv, module_activation):
        super(SageConvAndActivation, self).__init__()
        self.module_sageconv = module_sageconv
        self.module_activation = module_activation

    def forward(self, x, edge_index):
        return self.module_activation(self.module_sageconv(x, edge_index))
class SAGE(GrModuleWithLayeredEval):
    def __init__(self, list_dim_hidden, kwargs_sageconv, *args, **kwargs):
        '''

        :param list_dimhidden:
        :param kwargs_sageconv: kwargs of the `SageConv` layers other than `in_channels` and `out_channels`
        :param args:
        :param kwargs:
        '''
        super().__init__(*args, **kwargs)
        self.list_dim_hidden = list_dim_hidden
        self.list_dim = [self.dim_input] + list_dim_hidden + [self.dim_output]
        #make self.list_modules
        list_modules = []
        for l in range(len(self.list_dim) - 1):
            if l != len(self.list_dim) - 2:
                list_modules.append(
                    SageConvAndActivation(
                        module_sageconv=pyg.nn.SAGEConv(
                            **{**{'in_channels':self.list_dim[l],
                                  'out_channels':self.list_dim[l + 1]},
                               **kwargs_sageconv}
                        ),
                        module_activation=nn.ReLU()
                    )
                )
            else:
                list_modules.append(
                    pyg.nn.SAGEConv(
                        **{**{'in_channels': self.list_dim[l],
                              'out_channels': self.list_dim[l + 1]},
                           **kwargs_sageconv}
                    )
                )
        self.list_modules = nn.ModuleList(list_modules)




