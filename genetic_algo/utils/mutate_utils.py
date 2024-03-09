# import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubik54 import Move
import random

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


def prevent_moves(move_sequence, last_move, allowable_moves):
    allowed_moves = list(Move)  # Start with all moves allowed
    
    allowed_moves.remove(inverse_moves[last_move])
    if len(move_sequence) > 1 and move_sequence[-2] == last_move:
        allowed_moves.remove(last_move)
    return allowed_moves

# Prevention algorithm function
def get_allowed_mutations(move_sequence):    
    
    allowed_moves = list(Move)  # Start with all moves allowed

    if not move_sequence or move_sequence[-1] == Move.N:
        allowed_moves.remove(Move.N)  # Exclude the null move
        # If the sequence is empty or ends with a null move, all moves are allowed
        return allowed_moves
    
    last_move = move_sequence[-1]
    allowed_moves = prevent_moves(move_sequence, last_move, allowed_moves)
    return allowed_moves


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