from cube import Cube, load_model, Move, get_allowed_moves, inverse_moves, device
import numpy as np
import torch
import time
from collections import namedtuple
from heapq import heappush, heappop
from typing import Union
import random

nnet = load_model()
nnet1 = load_model("reversed")

@ torch.no_grad()
def compute_fitness(states):
    # Convert list of states to a NumPy array (if not already an array)
    states_np = np.array(states)
    # Convert NumPy array to a PyTorch tensor
    input_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
    # Compute output in a single batch
    outputs = nnet(input_tensor)
    fitness_scores = outputs.detach().cpu().numpy()
    torch.cuda.empty_cache()  # Clear CUDA cache here
    return fitness_scores

@ torch.no_grad()
def compute_prob(states):
    # Convert list of states to a NumPy array (if not already an array)
    states_np = np.array(states)
    # Convert NumPy array to a PyTorch tensor
    input_tensor = torch.tensor(states_np).to(device)
    # Compute output in a single batch
    outputs = nnet1(input_tensor)
    
    batch_p1 = torch.nn.functional.softmax(outputs, dim=1)
    batch_p1 = batch_p1.detach().cpu().numpy()
    torch.cuda.empty_cache()  # Clear CUDA cache here
    return batch_p1

# test if the admissible heuristic is correct
def test_admissible():
    for i in range(10000):
        cube = Cube()
        random_scramble_number = random.randint(2, 5)
        _, moves = cube.randomize_n(random_scramble_number)
        last_reversed_move = inverse_moves[moves[-1]]
        
        # get probs
        probs = compute_prob([cube.state])
        probs_of_last_move = probs[0][last_reversed_move]
        
        # do last inverse move
        cube.move(last_reversed_move)
        # get fitness
        fitness = compute_fitness([cube.state])
        
        assert fitness[0][0] - 0.3 * probs_of_last_move < random_scramble_number - 1, f"Failed at {i}th test"
        
        print(f"Passed {i}th test")
        
        
def test_original_admissible():
    for i in range(10000):
        cube = Cube()
        random_scramble_number = random.randint(2, 5)
        _, moves = cube.randomize_n(random_scramble_number)
        
        fitness = compute_fitness([cube.state])[0][0]
        
        print(fitness, random_scramble_number)
        
        assert fitness < random_scramble_number, f"Failed at {i}th test"
        
        print(f"Passed {i}th test")
        
if __name__ == "__main__":
    # test_admissible()
    # # test_original_admissible()
    
    # print("All tests passed!")
    
    temp = Cube()
    
    scramble_str, moves = (temp.randomize_n(13))
    print(scramble_str)
    
    probs = compute_prob([temp.state])[0]
    print(probs)
    
    probs = probs[inverse_moves[moves[-1]]]
    
    temp.move(inverse_moves[moves[-1]])
    
    fitness = compute_fitness([temp.state])[0][0]
    
    print(fitness)
    print(probs)
    print(fitness - probs)