import torch
import os
from typing import List, Optional
from collections import OrderedDict
from torch import nn
import re
from a_model import ResnetModel
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def state_to_nnet_input(states: List[Cube3State]) -> List[np.ndarray]:
    states_np = np.stack([state.colors for state in states], axis=0)

    representation_np: np.ndarray = states_np / (self.cube_len ** 2)
    representation_np: np.ndarray = representation_np.astype(self.dtype)

    representation: List[np.ndarray] = [representation_np]

    return representation

# loading nnet
def load_nnet(model_file: str, nnet: nn.Module) -> nn.Module:
    # get state dict

    state_dict = torch.load(model_file, map_location=device)

    # remove module prefix
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


state_dim: int = (3 ** 2) * 6
saved_model = "../saved_models/model_state_dict.pt"
model = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)

# test loading nnet
nnet = load_nnet(saved_model, model)

nnet.eval()

# def load_heuristic_fn(nnet_dir: str, device: torch.device, on_gpu: bool, nnet: nn.Module, env: Environment,
#                       clip_zero: bool = False, gpu_num: int = -1, batch_size: Optional[int] = None):
#     if (gpu_num >= 0) and on_gpu:
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)

#     model_file = "%s/model_state_dict.pt" % nnet_dir

#     nnet = load_nnet(model_file, nnet, device=device)
#     nnet.eval()
#     nnet.to(device)
#     if on_gpu:
#         nnet = nn.DataParallel(nnet)

#     heuristic_fn = get_heuristic_fn(nnet, device, env, clip_zero=clip_zero, batch_size=batch_size)

#     return heuristic_fn

