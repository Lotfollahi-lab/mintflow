
import torch
import torch.utils.data
from torchdyn.core import NeuralODE


@torch.no_grad()
def func_feed_x_to_module(module_input, x, batch_size):
    '''
    Feeds a (potentially) large tensor `x` to a module `module_input`.
    :param module_input:
    :param x: tensor of shape [NxD].
    :return:
    '''
    assert(not isinstance(module_input, NeuralODE))
    assert(len(x.size()) == 2)
    ds = torch.utils.data.TensorDataset(
        x,
        torch.tensor(range(x.size()[0]))
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    dict_n_to_output = {}
    for _, data in enumerate(dl):
        netout = module_input(data[0])
        for n_local, n_global in enumerate(data[1].tolist()):
            dict_n_to_output[n_global] = netout[n_local]
    assert(
        set(dict_n_to_output.keys()) == set(range(x.size()[0]))
    )
    ten_toret = torch.stack(
        [dict_n_to_output[n] for n in range(x.size()[0])],
        0
    )
    return ten_toret


@torch.no_grad()
def func_feed_x_to_neuralODEmodule(module_input, x, batch_size, t_span):
    '''
    Feeds a (potentially) large tensor `x` to a neural ODE module `module_input`.
    :param module_input:
    :param x: tensor of shape [NxD].
    :return:
    '''
    assert(isinstance(module_input, NeuralODE))
    assert(len(x.size()) == 2)
    ds = torch.utils.data.TensorDataset(
        x,
        torch.tensor(range(x.size()[0]))
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    dict_n_to_output = {}
    '''
    recall the output from neuralODE module is as follows
    - output[0]: is the t_range.
    - output[1]: is of shape [len(t_range), N, D].
    '''
    assert(module_input.return_t_eval)
    for _, data in enumerate(dl):
        _, netout = module_input(data[0], t_span)
        netout = netout[-1, :, :] #to pick the last solution
        for n_local, n_global in enumerate(data[1].tolist()):
            dict_n_to_output[n_global] = netout[n_local]
    assert(
        set(dict_n_to_output.keys()) == set(range(x.size()[0]))
    )
    ten_toret = torch.stack(
        [dict_n_to_output[n] for n in range(x.size()[0])],
        0
    )
    return ten_toret