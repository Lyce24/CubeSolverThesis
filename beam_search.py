"""
This module provides a beam search algorithm for finding a solution path in a given environment.
The `beam_search` function in this file is designed for readability and reproducibility, and it may not be the most speed-optimized implementation.
"""

import time
import numpy as np
from copy import deepcopy
from contextlib import nullcontext
import torch
            
            
if __name__ == "__main__":
    
    # from environments import Cube3
    
    # import torch
    
    # model = torch.jit.load("models/cube3.pth")
    
    # env = Cube3()
    
    # scramble = ['B', "U'", 'R', 'U', 'U', 'R', 'U', 'D', 'R', 'F', 'R', 'B', "R'", "F'", "U'", "R'", "F'", "F'", 'U', "R'", "F'", 'D']

    # env.reset()
    # env.apply_scramble(scramble)
    
    # print(beam_search(env, model))
    
    from rubik54 import Cube, Move
    import torch
    from utils.cube_model import ResnetModel
    from collections import OrderedDict
    import re
    
    import random
    import numpy as np

 
    model = torch.jit.load("saved_models/cube3.pth")
    
    
    model.eval()
    
    test_str = "U F U R U F D L U U F B"

    cube = Cube()
    cube.move_list(cube.convert_move(test_str))

  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = cube.convert_res_input()
    
    with torch.no_grad():
        x = torch.from_numpy(x).to(device)
        
        p = model(x)
        p = torch.nn.functional.softmax(p, dim=-1)
        p = p.detach().cpu().numpy()
        print(p)
    
    # load model
    def load_model():

        state_dim = 54
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True).to(device)
        model = "saved_models/phase2.pt"

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
        return nnet
    
    nnet = load_model()
    
    input_state = cube.convert_res_input()
    input_tensor = torch.tensor(input_state, dtype=torch.float32).to(device)
    output = nnet(input_tensor)

    print(output.item())
    