

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