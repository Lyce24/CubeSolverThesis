import torch.nn as nn
import torch.nn.functional as F

# 6, 5000, 1000, 4, 1,
class ResnetModel(nn.Module):
    def __init__(self, state_dim: int = 54, one_hot_depth: int = 6, h1_dim: int = 5000, resnet_dim: int = 1000, num_resnet_blocks: int = 4,
                 out_dim: int = 1, batch_norm: bool = False):
        
        # state_dim, 6, 5000, 1000, 4, 1, True)
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
  
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

class MLPModel(nn.Module):
    def __init__(self, input_size = 324):
        super().__init__()
        self.input_size = input_size
        self.body = nn.Sequential(
            nn.Linear(self.input_size, 1056),
            nn.ReLU(),
            nn.Linear(1056, 3888),
            nn.ReLU(),
            nn.Linear(3888, 1104),
            nn.ReLU(),
            nn.Linear(1104, 576),
            nn.ReLU(),
            nn.Linear(576,1)
        )

        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.detach().zero_()

    def forward(self, batch):
        x = batch.reshape((-1, self.input_size))
        body_out = self.body(x)
        return body_out

    def clone(self):
        new_state_dict = {}
        for kw, v in self.state_dict().items():
            new_state_dict[kw] = v.clone()
        new_net = MLPModel(self.input_size)
        new_net.load_state_dict(new_state_dict)
        return new_net