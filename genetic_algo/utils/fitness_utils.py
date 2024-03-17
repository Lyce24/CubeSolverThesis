# import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubik54 import Cube, Move
import torch
from utils.cube_model import MLPModel, ResnetModel
from collections import OrderedDict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

single_move = [
    Move.U1, Move.U3, Move.R1, Move.R3, Move.F1, Move.F3, Move.D1, Move.D3, Move.L1, Move.L3, Move.B1, Move.B3
]

double_move_to_single_moves = {
    Move.U2 : Move.U1,
    Move.R2 : Move.R1,
    Move.F2 : Move.F1,
    Move.D2 : Move.D1,
    Move.L2 : Move.L1,
    Move.B2 : Move.B1
}


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
    Move.B3 : Move.B1,
    Move.U2 : Move.U2,
    Move.R2 : Move.R2,
    Move.F2 : Move.F2,
    Move.D2 : Move.D2,
    Move.L2 : Move.L2,
    Move.B2 : Move.B2
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
    Move.B3 : "FB",
    Move.U2 : "UD",
    Move.R2 : "LR",
    Move.F2 : "FB",
    Move.D2 : "UD",
    Move.L2 : "LR",
    Move.B2 : "FB"
}

# Helper functions not provided:
def get_move_group(move):
    # Return the group number for the given move
    return groups[move]

def filter(individual):
    # record null move positions
    null_moves = []
    for i in range(len(individual)):
        if individual[i] == Move.N:
            null_moves.append(i)
    
    # remove the null moves
    individual = [move for move in individual if move != Move.N]

    score = 1

    # Simplify the sequence by removing redundant moves
    i = 0
    while i < len(individual) - 1:

        # Identify the group of the current move
        first_move = individual[i]
        last_group = get_move_group(first_move)
        
        # Initialize the subsequence with the current move
        subsequence = [first_move]
        
        # Use j to find the length of the subsequence
        j = i + 1
        while j < len(individual) and get_move_group(individual[j]) == last_group:
            subsequence.append(individual[j])
            j += 1
        
        # Calculate subsequence length
        subsequence_length = j - i
            
        
        if subsequence_length == 1:
            i += 1
            continue
            
        # Remove redundant moves
        pair_map = {}
    
        if last_group == "UD":
            pair_map = {Move.U1 : 0, Move.D1 : 0}
        elif last_group == "LR":
            pair_map = {Move.L1 : 0, Move.R1 : 0}
        else:
            pair_map = {Move.F1 : 0, Move.B1 : 0}
            
        for move in subsequence:
            if move in pair_map:
                pair_map[move] += 1
            if move in double_move_to_single_moves and double_move_to_single_moves[move] in pair_map:
                pair_map[double_move_to_single_moves[move]] += 2
            if inverse_moves[move] in pair_map:
                pair_map[inverse_moves[move]] -= 1
                    
        temp = 0
        # Insert the first one or two moves at the beginning of the subsequence
        # and make the rest of the subsequence null moves.
        for move in pair_map:
            if pair_map[move] % 4 != 0:
                temp += 1
        
        score += (subsequence_length - temp)
        
        # Skip over the processed subsequence in the next iteration
        i += subsequence_length
        
    return score


if __name__ == "__main__":
    # Test the filter function
    individual = [Move.U1, Move.F1]
    print(filter(individual)) 