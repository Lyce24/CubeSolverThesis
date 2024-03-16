# import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubik54 import Cube, Move
import torch
from utils.cube_model import MLPModel, ResnetModel
from collections import OrderedDict
import re
from utils.selection_utils import phase2_filters
from concurrent.futures import ThreadPoolExecutor, as_completed


inverse_moves = {
    Move.U1 : Move.U3,
    Move.R1 : Move.R3,
    Move.F1 : Move.F3,
    Move.D1 : Move.D3,
    Move.L1 : Move.L3,
    Move.B1 : Move.B3,
    Move.U3 : Move.U1,
    Move.R3 : Move.R1,
    Move.F3 : Move.F1,
    Move.D3 : Move.D1,
    Move.L3 : Move.L1,
    Move.B3 : Move.B1
}

groups = {
    Move.U1 : "UD",
    Move.R1 : "LR",
    Move.F1 : "FB",
    Move.D1 : "UD",
    Move.L1 : "LR",
    Move.B1 : "FB",
    Move.U3 : "UD",
    Move.R3 : "LR",
    Move.F3 : "FB",
    Move.D3 : "UD",
    Move.L3 : "LR",
    Move.B3 : "FB"
}

# Helper functions not provided:
def get_move_group(move):
    # Return the group number for the given move
    return groups[move]

def apply_filter_1(moves):
    # For simplicity, we assume that each move has an identifier that indicates its group
    last_group = get_move_group(moves[-1])
    subsequence_length = 1
    for move in reversed(moves[:-1]):
        if get_move_group(move) == last_group:
            subsequence_length += 1
        else:
            break
    # If the subsequence is entirely from one group, penalize it. Also return the subsequence for filter 2
    return subsequence_length if subsequence_length > 1 else 0, moves[-subsequence_length:]

def apply_filter_2(moves):
    # Split the subsequence into sets of moves for one face
    
    last_group = get_move_group(moves[-1])
    
    if last_group == "UD":
        pair_map = {Move.U1 : Move.U3, Move.U3 : Move.U1, Move.D1 : Move.D3, Move.D3 : Move.D1}
    elif last_group == "LR":
        pair_map = {Move.L1 : Move.L3, Move.L3 : Move.L1, Move.R1 : Move.R3, Move.R3 : Move.R1}
    else:
        pair_map = {Move.F1 : Move.F3, Move.F3 : Move.F1, Move.B1 : Move.B3, Move.B3 : Move.B1}
    
    
    # Dictionary to count occurrences of each number
    count_map = {}
    
    for i in moves:
        if i in pair_map:  # Only count if the number is part of a pair
            count_map[i] = count_map.get(i, 0) + 1
    
    # Calculate pairs
    total_pairs = 0
    counted = set()  # Keep track of counted pairs to avoid double counting
    for num in count_map:
        if num not in counted and pair_map[num] in count_map:
            pair_num = pair_map[num]
            pairs = min(count_map[num], count_map[pair_num])
            total_pairs += pairs
            counted.add(num)
            counted.add(pair_num)
    
    return total_pairs


def phase2_filters(moves):
    filter_count = 0
    
    temp = [move for move in moves if move != Move.N]
    
    # Filter 1: Longest subsequence in the end of the child sequence from one group
    filter_1, subsequence = apply_filter_1(temp)
    filter_count += filter_1
    
    # Filter 2: Penalize subsequences that neutralize each other
    filter_count += apply_filter_2(subsequence)
    
    
    return 1 if filter_count == 0 else filter_count


# Genetic Algorithm Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

inverse_moves = {
    Move.U1 : Move.U3,
    Move.R1 : Move.R3,
    Move.F1 : Move.F3,
    Move.D1 : Move.D3,
    Move.L1 : Move.L3,
    Move.B1 : Move.B3,
    Move.U3 : Move.U1,
    Move.R3 : Move.R1,
    Move.F3 : Move.F1,
    Move.D3 : Move.D1,
    Move.L3 : Move.L1,
    Move.B3 : Move.B1
}

# load model
def load_model(model_type):
    if model_type == "mlp":
        nnet = MLPModel(324).to(device)
        model = "saved_models/model102.dat"
        nnet.load_state_dict(torch.load(model, map_location=device))
        nnet.eval()
        return nnet
    
    elif model_type == "resnet":
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
    else:
        raise ValueError("Invalid model type")

def compute_fitness(scrambled_str, individual, model, nnet):
    cube = Cube()
    cube.from_string(scrambled_str)
    cube.move_list(individual)
    
    """Compute the fitness of an individual solution."""
    if cube.is_solved():
        return 100
    else:
        if model == "mlp":
            input_state= cube.convert_mlp_input()
        elif model == "resnet":
            input_state = cube.convert_res_input()
        else:
            raise ValueError("Invalid model type")
        input_tensor = torch.tensor(input_state, dtype=torch.float32).to(device)
        output = nnet(input_tensor)
        if model == "mlp":
            return output.item()
        return -output.item()
    
def compute_fitness_v1(scrambled_str, individual):
    """Compute the fitness of an individual solution."""
    cube = Cube()
    cube.from_string(scrambled_str)
    # Initialize a solved cube
    solved_cube = Cube()
    # Apply the sequence of moves from the individual to a new cube
    cube.move_list(individual)
    
    # Compare each subface to the solved state
    correct_subfaces = sum(1 for i in range(54) if cube.f[i] == solved_cube.f[i])
    return correct_subfaces
    
def compute_fitness_v2(scrambled_str, individual):
    """Compute the fitness of an individual solution."""
    
    filters = phase2_filters(individual)
    
    cube = Cube()
    cube.from_string(scrambled_str)
    # Initialize a solved cube
    solved_cube = Cube()
    # Apply the sequence of moves from the individual to a new cube
    cube.move_list(individual)
    
    # Compare each subface to the solved state
    correct_subfaces = sum(1 for i in range(54) if cube.f[i] == solved_cube.f[i])
    return correct_subfaces/filters
    

def compute_fitness_phase1(scrambled_str, individual):
    """Compute the fitness of an individual solution."""
        
    cube = Cube()
    cube.from_string(scrambled_str)
    cube.move_list(individual)
    
    co, eo = cube.get_phase1_state()
    
    if cube.check_phase1_solved():
        return 20
    
    # calculate how many pieces is in the correct position
    total = 0
    for i in range(8):
        if co[i] == 0:
            total += 1
    for i in range(12):
        if eo[i] == 0:
            total += 1
            
    return total

    
# Parallel execution of compute_fitness
def compute_fitness_in_parallel(scrambled_str, individuals, model, nnet, mode = "v1"):
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        if mode == "drl":
            future_to_individual = {executor.submit(compute_fitness, scrambled_str, ind, model, nnet): ind for ind in individuals}
        elif mode == "v1":
            future_to_individual = {executor.submit(compute_fitness_v1, scrambled_str, ind): ind for ind in individuals}
        elif mode == "v2":
            future_to_individual = {executor.submit(compute_fitness_v2, scrambled_str, ind): ind for ind in individuals}
        elif mode == "phase1":
            future_to_individual = {executor.submit(compute_fitness_phase1, scrambled_str, ind): ind for ind in individuals}
        
        for future in as_completed(future_to_individual):
            individual = future_to_individual[future]
            try:
                fitness = future.result()
                results.append((individual, fitness))
            except Exception as exc:
                print('%r generated an exception: %s' % (individual, exc))
    return results