

'''
Auxiliary modules are moved to this file, so the inflow checkpoint dumped in CLI can be unpickled.
'''


import torch

class ModuleListandConcatHeads(torch.nn.Module):
    def __init__(self, list_modules):
        super(ModuleListandConcatHeads, self).__init__()

        assert isinstance(list_modules, list)
        for u in list_modules:
            assert isinstance(u, torch.nn.Module)

        self.list_modules = torch.nn.ModuleList(list_modules)

    def forward(self, x):
        return torch.cat(
            [m(x) for m in self.list_modules],
            1
        )

