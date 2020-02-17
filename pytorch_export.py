import readline
import rlcompleter
readline.parse_and_bind('tab: complete')

import torch
import torch.nn as nn
import numpy as np

import poptorch


class Layer(nn.Module):
    def __init__(self, in_size, out_size):
        super(Layer, self).__init__()

        self.net = nn.Linear(in_size, out_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.net(x)
        return self.relu(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = Layer(10, 10)
        self.layer2 = Layer(10, 5)

    def forward(self, x):
        # poptorch.pipeline_stage(0)
        # poptorch.virtual_graph(0)
        x = self.layer1(x)
        # poptorch.pipeline_stage(1)
        # poptorch.virtual_graph(1)
        # x = self.layer2(x)
        return x


net = Net()
net.layer1.vgraph = 0
net.layer2.vgraph = 1
# Using trace to avoid dealing with if statements for now.
# n = torch.jit.script(net)
x = torch.ones(10)
n = torch.jit.trace(net, x)

n.save('foo.pt')
poptorch.transformPass('foo.pt', x)

# print(type(n.graph))
# poptorch.foo(n.graph)
