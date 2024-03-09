# import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubik54 import Move


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
    
    penalties = 0
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

