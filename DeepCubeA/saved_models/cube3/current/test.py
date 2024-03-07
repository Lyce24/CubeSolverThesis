import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import re
from collections import OrderedDict


class ResnetModel(nn.Module):
    def __init__(self, state_dim: int, one_hot_depth: int, h1_dim: int, resnet_dim: int, num_resnet_blocks: int,
                 out_dim: int, batch_norm: bool):
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        if one_hot_depth > 0:
            self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
        else:
            self.fc1 = nn.Linear(self.state_dim, h1_dim)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state_dim = 54
nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)
model = "model_state_dict.pt"

state_dict = torch.load(model, map_location=device)
# remove module prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

# set state dict
nnet.load_state_dict(new_state_dict)


# load model
nnet.eval()


"""
[array([[0, 0, 0, 0, 0, 0, 4, 4, 4, 1, 1, 1, 1, 1, 1, 5, 5, 5, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 4, 4, 4, 4, 4,
        4, 5, 5, 5, 5, 5, 5, 0, 0, 0]], dtype=uint8)]
"""
test_input = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 4, 4, 4, 4, 4,
        4, 5, 5, 5, 5, 5, 5, 4, 4, 4], dtype=np.uint8)

input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)
output = nnet(input_tensor)
print(output.item())
