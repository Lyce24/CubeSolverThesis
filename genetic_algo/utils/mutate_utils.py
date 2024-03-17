# import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubik54 import Move
import random

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

double_move = {
    Move.U1 : Move.U2,
    Move.R1 : Move.R2,
    Move.F1 : Move.F2,
    Move.D1 : Move.D2,
    Move.L1 : Move.L2,
    Move.B1 : Move.B2
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

def get_last_move(move_sequence):
    # select the last move in the sequence other than the null move. Return both the last move and the index to it
    # Otherwise, return the null move
    for move in (reversed(move_sequence)):
        if move != Move.N:
            return move
    return Move.N

def get_first_move(move_sequence):
    # select the first move in the sequence other than the null move. Return both the first move and the index to it
    # Otherwise, return the null move
    for move in (move_sequence):
        if move != Move.N:
            return move
    return Move.N

# Helper functions not provided:
def get_move_group(move):
    # Return the group number for the given move
    return groups[move]


def prevent_moves_pre(move_sequence, last_move, allowed_moves):

    temp = [move for move in move_sequence if move != Move.N]
    subsequence = []
    last_group = get_move_group(last_move)
    for move in reversed(temp):
        if get_move_group(move) == last_group:
            subsequence.append(move)
        else:
            break
        
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
            
    for i in pair_map:
        if pair_map[i] % 4 == 3:
            if i in allowed_moves:
                allowed_moves.remove(i)
        elif pair_map[i] % 4 == 2:
            if double_move[i] in allowed_moves:
                allowed_moves.remove(double_move[i])
        elif pair_map[i] % 4 == 1:
            if inverse_moves[i] in allowed_moves:
                allowed_moves.remove(inverse_moves[i])

    return allowed_moves


def get_allowed_mutations_pre(move_sequence):
    
    allowed_moves = list(Move)  # Start with all moves allowed
    
    if not move_sequence:
        allowed_moves.remove(Move.N)  # Exclude the null move
        return allowed_moves
    
    last_move = get_last_move(move_sequence)
    
    if last_move == Move.N:
        allowed_moves.remove(Move.N)  # Exclude the null move
        return allowed_moves
    
    allowed_moves = prevent_moves_pre(move_sequence, last_move, allowed_moves)
    return allowed_moves

def prevent_moves_post(move_sequence, first_move, allowed_moves):
    
    temp = [move for move in move_sequence if move != Move.N]
    subsequence = []
    last_group = get_move_group(first_move)
    for move in temp:
        if get_move_group(move) == last_group:
            subsequence.append(move)
        else:
            break
        
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
            
    for i in pair_map:
        if pair_map[i] % 4 == 3:
            if i in allowed_moves:
                allowed_moves.remove(i)
        elif pair_map[i] % 4 == 2:
            if double_move[i] in allowed_moves:
                allowed_moves.remove(double_move[i])
        elif pair_map[i] % 4 == 1:
            if inverse_moves[i] in allowed_moves:
                allowed_moves.remove(inverse_moves[i])

    return allowed_moves


def get_allowed_mutations_post(move_sequence):
    
    allowed_moves = list(Move)  # Start with all moves allowed
    
    if not move_sequence:
        allowed_moves.remove(Move.N)  # Exclude the null move
        return allowed_moves
    
    first_move = get_first_move(move_sequence)
    
    if first_move == Move.N:
        allowed_moves.remove(Move.N)  # Exclude the null move
        return allowed_moves
    
    allowed_moves = prevent_moves_post(move_sequence, first_move, allowed_moves)
    return allowed_moves
        
# Prevention algorithm function
def get_allowed_mutations(move_sequence1, move_sequence2): 
    # Get the allowed mutations for the first move group
    allowed_moves1 = get_allowed_mutations_pre(move_sequence1)
    # Get the allowed mutations for the second move group
    allowed_moves2 = get_allowed_mutations_post(move_sequence2)
    # Return the intersection of the two sets
    return list(set(allowed_moves1) & set(allowed_moves2))



def single_point_mutate(individual):
    # Randomly choose a gene (mutation point) to mutate
    gene_to_mutate = random.randint(0, len(individual) - 1)
    
    # Get the moves performed so far up to the mutation point
    move_sequence_up_to_point = individual[:gene_to_mutate]
    
    # Get the list of allowed moves for the mutation based on the last move group
    allowed_moves = get_allowed_mutations(move_sequence_up_to_point)
 
    # Apply a valid mutation from the allowed moves
    individual[gene_to_mutate] = random.choice(allowed_moves)

    return individual

def random_mutate(individual):
    mutation_type = random.choice(['swap', 'scramble', 'invert', 'insert', 'delete'])
    
    if mutation_type == 'swap':
        # Swap two moves in the sequence
        if len(individual) > 1:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    
    elif mutation_type == 'scramble':
        # Scramble a subset of the sequence
        if len(individual) > 1:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end] = random.sample(individual[start:end], len(individual[start:end]))
    
    elif mutation_type == 'invert':
        # Invert a subset of the sequence
        if len(individual) > 1:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end] = individual[start:end][::-1]
    
    elif mutation_type == 'insert':
        # Insert a random move at a random position
        count = random.randint(1, 4)
        
        for _ in range(count):
            idx = random.randint(0, len(individual))
            move = random.choice(list(Move))
            individual.insert(idx, move)
    
    elif mutation_type == 'delete':
        # Delete a random move
        if len(individual) > 1:
            idx = random.randint(0, len(individual) - 1)
            del individual[idx]
    
    return individual


def simplify_individual(individual):
    # record null move positions
    null_moves = []
    for i in range(len(individual)):
        if individual[i] == Move.N:
            null_moves.append(i)
    
    # remove the null moves
    individual = [move for move in individual if move != Move.N]

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
                    
        # Insert the first one or two moves at the beginning of the subsequence
        # and make the rest of the subsequence null moves.
        for index, move in enumerate(pair_map):
            if pair_map[move] % 4 == 1:
                individual[i + index] = move
            elif pair_map[move] % 4 == 2:
                individual[i + index] = double_move[move]
            elif pair_map[move] % 4 == 3:
                individual[i + index] = inverse_moves[move]
            elif pair_map[move] % 4 == 0:
                individual[i + index] = Move.N
                    
        rest_of_subsequence = subsequence_length - 2
        
        for index in range(rest_of_subsequence):
            individual[i + 2 + index] = Move.N
        
        
        # Skip over the processed subsequence in the next iteration
        i += subsequence_length
        
 
            
    # reinsert the null moves
    for i in null_moves:
        individual.insert(i, Move.N)
        
    return individual

if __name__ == "__main__":
    # Test the mutation functions
    
    length = 0
    
    for i in range(1000):
        individual = [random.choice(list(Move)) for _ in range(23)]
        print(individual)
        new_individual = simplify_individual(individual)
        print(len(new_individual) == 23)
        print(new_individual)
        temp = [move for move in new_individual if move != Move.N]
        length += len(temp)
    
    print(length / 1000)