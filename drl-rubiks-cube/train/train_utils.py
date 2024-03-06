import numpy as np
import torch
from random import randint
from keras.utils.np_utils import to_categorical
from cube import get_solved,idxs

'''
batch_size=1000
scramble_depth=30
report_batches=100
include_first=True
weight_type=learn_first_close
'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.nn
from torch import nn


class Net(nn.Module):
    def __init__(self, input_size):
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
            if isinstance(m,torch.nn.Linear):
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
        new_net = Net(self.input_size)
        new_net.load_state_dict(new_state_dict)
        return new_net
    
nnet = Net(54*6).to(device)

def generate_states_by_ADI(conf, nnet, device):
    # times = 10000//30
    times = 1
    states, depths = generate_move_seq(times,1, True)
    print(states.shape, depths.shape)
    print(states)
    print(depths)
    val_targets = explore_states(states,nnet,device).to(device)
    states = torch.from_numpy(to_categorical(states)).to(device)
    weights = get_weigths(depths,"learn_first_close").to(device)
    return states.detach(),val_targets.detach(),weights.detach()


def generate_move_seq(times,scramble_depth,include_first):
    seq = np.random.randint(12,size=times*scramble_depth)
    for t in range(times):
        for idx in range(1,scramble_depth):
            while True:
                if (seq[idx+t*scramble_depth] + 6) % 12 == seq[idx+t*scramble_depth - 1]:
                    seq[idx+t*scramble_depth] = randint(0, 11)
                elif idx > 1 and (seq[idx+t*scramble_depth] == seq[idx+t*scramble_depth - 2] and seq[idx+t*scramble_depth] == seq[idx+t*scramble_depth - 1]):
                    seq[idx+t*scramble_depth] = randint(0, 11)
                else:
                    break

    depths = np.ones(times*scramble_depth,dtype=np.float32)
    state = get_solved()
    states = np.repeat(np.expand_dims(state, axis=0), repeats=times*scramble_depth, axis=0)

    for time in range(times):
        states[time*scramble_depth] = states[time*scramble_depth][idxs[seq[time*scramble_depth]]]
        depths[time*scramble_depth] = 1
        for depth in range(scramble_depth-1):
            states[time * scramble_depth + depth + 1] = states[time * scramble_depth + depth][idxs[seq[time * scramble_depth + depth + 1]]]
            depths[time * scramble_depth + depth + 1] = depth+2

    if include_first:
        solved_states = np.repeat(np.expand_dims(state, axis=0), repeats=times, axis=0)
        solved_depths = np.ones(times,dtype=np.float32)
        states = np.concatenate([states,solved_states])
        depths = np.concatenate([depths,solved_depths])
    shuffle_idxs = np.arange(depths.shape[0])
    np.random.shuffle(shuffle_idxs)
    states = states[shuffle_idxs]
    depths = depths[shuffle_idxs]
    return states,depths


def generate_move_seq_random(times,scramble_depth,include_first):
    seq = np.random.randint(12,size=times*scramble_depth)
    depths = np.ones(times*scramble_depth,dtype=np.float32)
    state = get_solved()
    states = np.repeat(np.expand_dims(state, axis=0), repeats=times*scramble_depth, axis=0)

    for time in range(times):
        states[time*scramble_depth] = states[time*scramble_depth][idxs[seq[time*scramble_depth]]]
        depths[time*scramble_depth] = 1
        for depth in range(scramble_depth-1):
            states[time * scramble_depth + depth + 1] = states[time * scramble_depth + depth][idxs[seq[time * scramble_depth + depth + 1]]]
            depths[time * scramble_depth + depth + 1] = depth+2

    if include_first:
        solved_states = np.repeat(np.expand_dims(state, axis=0), repeats=times, axis=0)
        solved_depths = np.ones(times,dtype=np.float32)
        states = np.concatenate([states,solved_states])
        depths = np.concatenate([depths,solved_depths])
    shuffle_idxs = np.arange(depths.shape[0])
    np.random.shuffle(shuffle_idxs)
    states = states[shuffle_idxs]
    depths = depths[shuffle_idxs]
    return states,depths


def explore_states(states,net,device):
    solved = get_solved()
    substates = np.repeat(np.expand_dims(states, axis=1), repeats=12, axis=1)
    substates = np.squeeze(substates[:, ::12])[:, idxs]

    solved_states = (states == solved).all(axis=states.ndim-1)
    solved_substates = (substates == solved).all(axis=substates.ndim-1)
    rewards_substates = np.ones(solved_substates.shape,dtype=np.float32)
    rewards_substates[~solved_substates] = -1
    rewards_substates = torch.from_numpy(rewards_substates).to(device)
    substates = torch.from_numpy(to_categorical(substates)).to(device)
    values = net(substates)
    values = values.view(-1, 12) + rewards_substates.view(-1, 12)
    values = torch.max(values, 1)[0]
    values[solved_states] = 0

    return values


def update_generator(generator, nnet, tau, device):
    generator_params, net_params = generator.state_dict(), nnet.state_dict()
    new_generator_params = dict(generator_params)
    for name, param in net_params.items():
        new_generator_params[name].data.copy_(
            tau * param.data.to(device) + (1 - tau) * new_generator_params[name].data.to(device)
        )
    generator.load_state_dict(new_generator_params)
    return generator.to(device)


def get_weigths(depths, type):
    if type == "learn_first_close":
        return 1/torch.from_numpy(depths)
    elif type == "lightweight":
        return 1/(torch.from_numpy(depths)/2)
    else:
        return torch.from_numpy(depths)

x, y, weight = generate_states_by_ADI(None,nnet,device)
print(x.shape, y.shape, weight.shape)

# print head of the data
print(x[:5])
print(y[:5])
print(weight[:5])

