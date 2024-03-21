"""
This module provides a beam search algorithm for finding a solution path in a given environment.
The `beam_search` function in this file is designed for readability and reproducibility, and it may not be the most speed-optimized implementation.
"""

import time
import torch
from rubik54 import Cube
from utils.cube_model import ResnetModel
from collections import OrderedDict
import re
from utils.test_utils import get_allowed_mutations_pre
import numpy as np

            
"""
This module provides a beam search algorithm for finding a solution path in a given environment.
The `beam_search` function in this file is designed for readability and reproducibility, and it may not be the most speed-optimized implementation.
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load model
def load_model():

        state_dim = 54
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True).to(device)
        model = "saved_models/model_state_dict.pt"

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

class Beam_Node:
    def __init__(self, cube, moves, fitness, is_solved):
        self.cube = cube
        self.moves = moves
        self.fitness = fitness
        self.is_solved = is_solved
        
def compute_fitness(states):
    # Convert list of states to a NumPy array (if not already an array)
    states_np = np.array(states)
    # Convert NumPy array to a PyTorch tensor
    input_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
    # Compute output in a single batch
    outputs = nnet(input_tensor)
    return outputs.detach().cpu().numpy()

def generate_new_generation(generation: list[Beam_Node], prevention):
    new_generation = []
    nodes_searched = 0
    batch_states = []
    batch_info = []  # To keep track of the corresponding cube and moves

    for i in generation:
        if not prevention:
            allowed_moves = Cube().get_possible_moves()
        else:
            allowed_moves = get_allowed_mutations_pre(i.moves)
            
        for move in allowed_moves:
            new_moves = i.moves + [move]
            tempcube = Cube()
            tempcube.from_string(i.cube)
            tempcube.move(move)
            state = tempcube.convert_res_input()
            
            if tempcube.is_solved():
                nodes_searched += 1
                new_generation.append(Beam_Node(tempcube.to_string(), new_moves, 0, True))
                continue

            batch_states.append(state)
            batch_info.append((tempcube, new_moves))
            nodes_searched += 1

    # Convert batch_states to numpy array and compute fitness in one go
    batch_states_np = np.array(batch_states)
    fitness_scores = compute_fitness(batch_states_np)

    # Create Beam_Node instances using the fitness scores
    for (tempcube, new_moves), fitness in zip(batch_info, fitness_scores):
        new_generation.append(Beam_Node(tempcube.to_string(), new_moves, fitness, False))

    # Sort new generation based on fitness
    new_generation.sort(key=lambda x: x.fitness)
    return new_generation, nodes_searched


def beam_search(scrambled_cube : Cube, beam_width = 1024, max_depth = 100, prevention = True) -> dict:
    
    root = Beam_Node(scrambled_cube.to_string(), [], compute_fitness(scrambled_cube.convert_res_input()), cube.is_solved())
    generation = [root]
    start_time = time.time()
    node_searched = 0
    
    for depth in range(max_depth + 1):
        print(f"Depth: {depth}, len(generation): {len(generation)}")
        for i in generation:
            if i.is_solved:
                return {"success" : True, "solutions": i.moves, "num_nodes": node_searched, "time_taken": time.time() - start_time}
    
        if depth == max_depth:
            return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}
        
        new_generation, searched_nodes = generate_new_generation(generation, prevention)
        node_searched += searched_nodes
        generation = new_generation[:beam_width]
        print("Best fitness: ", generation[0].fitness)
        print("Best moves: ", generation[0].moves)
        
    return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}

            
if __name__ == "__main__":
    test_str = "U D' F B L' B' R L' D L' L' D F' U' R F L' B' F"

    cube = Cube()
    cube.move_list(cube.convert_move(test_str))
    
    print(beam_search(cube, 1000, 30, False))
    # convert list to numpy list
    
    # example = [cube.convert_res_input() for _ in range(10000)]
    # example = np.array(example)
    
    # start_time = time.time()
    # nnet(torch.tensor(example, dtype=torch.float32).to(device))
    # print(time.time() - start_time)