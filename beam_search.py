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
from utils.search_utils import get_allowed_moves
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
    
    tempcube = Cube()
    for node in generation:
        allowed_moves = get_allowed_moves(node.moves) if prevention else Cube().get_possible_moves()
            
        for move in allowed_moves:
            new_moves = node.moves + [move]
            tempcube = Cube()
            tempcube.from_string(node.cube)
            tempcube.move(move)
            state = tempcube.convert_res_input()
            
            if tempcube.is_solved():
                nodes_searched += 1
                new_generation.append(Beam_Node(tempcube.to_string(), new_moves, 0, True))
                return [Beam_Node(tempcube.to_string(), new_moves, 0, True)], nodes_searched, True

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
    return new_generation, nodes_searched, False

def beam_search(scrambled_cube : Cube, beam_width = 1024, max_depth = 100, prevention = True) -> dict:
    
    root = Beam_Node(scrambled_cube.to_string(), [], compute_fitness(scrambled_cube.convert_res_input()), cube.is_solved())
    generation = [root]
    start_time = time.time()
    node_searched = 0
    
    for depth in range(max_depth + 1):
              
        if depth == max_depth:
            return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}
                
        new_generation, searched_nodes, success = generate_new_generation(generation, prevention)
        
        if success:
            return {"success" : True, "solutions": new_generation[0].moves, "num_nodes": node_searched + searched_nodes, "time_taken": time.time() - start_time}
        
        node_searched += searched_nodes
        generation = new_generation[: int(beam_width * (1 + (2 * depth)/100))]
        
        
    raise Exception("Beam search failed to find a solution")
            
if __name__ == "__main__":
    test_str = "U D' F B L' B' R L' D L' L' D F' U' R F L' B' F"
    
    success = 0
    total_sol_length = 0
    total_nodes = 0
    total_time = 0

    for i in range(0, 100):
        print(f"Test {i + 1}")
        cube = Cube()
        cube.randomize_n(100)
        
        result = beam_search(cube, 2048, 40, True)
        if result["success"]:
            success += 1
            total_sol_length += len(result["solutions"])
            total_nodes += result["num_nodes"]
            total_time += result["time_taken"]
            print(f"Success: {success}, Sol Length: {len(result['solutions'])}, Num Nodes: {result['num_nodes']}, Time Taken: {result['time_taken']}")
    
    print(f"Success Rate: {success/100}, Avg Sol Length: {total_sol_length/success}, Avg Num Nodes: {total_nodes/success}, Avg Time Taken: {total_time/success}")