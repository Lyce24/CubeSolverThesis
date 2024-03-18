# import from parent directory
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubik54 import Move, phase2_move, Cube
import random
import torch
from utils.cube_model import ResnetModel
from collections import OrderedDict
import re
import numpy as np

# Genetic Algorithm Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

def get_allowed_mutations_phase2(move_sequence1, move_sequence2):
    """
    U, U', U2, D, D', D2, R2, L2, F2, B2
    """
    # Get the allowed mutations for the first move group
    allowed_moves1 = get_allowed_mutations_pre(move_sequence1)
    # Get the allowed mutations for the second move group
    allowed_moves2 = get_allowed_mutations_post(move_sequence2)
    
    # Return the intersection of the two sets
    return list(set(allowed_moves1) & set(allowed_moves2) & set(phase2_move))

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


def generate_individual(SEQUENCE_LENGTH, phase):
    """Generate a random individual solution."""
    if phase == 1:
        individual = [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]
    elif phase == 2:
        individual = [random.choice(phase2_move) for _ in range(SEQUENCE_LENGTH)]
    else:
        raise ValueError("Phase must be 1 or 2")
    return individual

def mutate(individual, phase, mutation_rate=None):
    if mutation_rate:
        # Randomly choose whether to mutate
        if random.random() > mutation_rate:
            return individual
    
    # Randomly choose a gene (mutation point) to mutate
    gene_to_mutate = random.randint(0, len(individual) - 1)
    
    # Get the moves performed so far up to the mutation point
    move_sequence1 = individual[:gene_to_mutate]
     
    # get the moves performed after the mutation point
    if gene_to_mutate < len(individual) - 1:
        move_sequence2 = individual[gene_to_mutate + 1:]
    else:
        move_sequence2 = []
    
    allowed_moves = []
    if phase == 1:
        # Get the list of allowed moves for the mutation based on the last move group
        allowed_moves = get_allowed_mutations(move_sequence1, move_sequence2)
        
    elif phase == 2:
        # Get the list of allowed moves for the mutation based on the last move group
        allowed_moves = get_allowed_mutations_phase2(move_sequence1, move_sequence2)
    
    else:
        raise ValueError("Phase must be 1 or 2")
    
    # Apply a valid mutation from the allowed moves
    new_individual = individual.copy()
    new_individual[gene_to_mutate] = random.choice(allowed_moves)

    return new_individual


def compute_fitness(scrambled_str, individual, phase, nnet = None):
    """Compute the fitness of an individual solution based on different phases and neural network."""

    cube = Cube()
    cube.from_string(scrambled_str)
    cube.move_list(individual)
    
    if phase == 1:
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
    elif phase == 2:
        """Compute the fitness of an individual solution."""
        if cube.is_solved():
            return 100
        elif nnet:
            input_state = cube.convert_res_input()

            input_tensor = torch.tensor(input_state, dtype=torch.float32).to(device)
            output = nnet(input_tensor)

            return -output.item()
        else:
            solved_cube = Cube()
            # Apply the sequence of moves from the individual to a new cube
            cube.move_list(individual)
            
            # Compare each subface to the solved state
            correct_subfaces = sum(1 for i in range(54) if cube.f[i] == solved_cube.f[i])
            return correct_subfaces
        
    else:
        raise ValueError("Phase must be 1 or 2")
    
def elitism(scored_population, elite_rate = 0.007):
    """Carry over the top individuals to the next generation."""
    scored_population.sort(key=lambda x: x[1], reverse=True)
    elite_size = int(len(scored_population) * elite_rate)
    elite_individuals = [individual for individual, _ in scored_population[:elite_size]]
    return elite_individuals

def kill_by_rank(scored_population, survival_rate=0.7):
    # Rank population by fitness
    scored_population.sort(key=lambda x: x[1], reverse=True)
    
    # Select survivors based on survival rate
    num_survivors = int(len(scored_population) * survival_rate)
    survivors = scored_population[:num_survivors]
    
    # Return survivors - first element of each tuple
    return [individual for individual, _ in survivors]

def boltzmann_selection(population, scored_population, temperature):
    # Extract fitness scores and normalize them
    fitness_scores = [score for _, score in scored_population]
    
    min_fitness = min(fitness_scores)
    if min_fitness < 0:
        offset = abs(min_fitness) + 1  # Ensure all scores are positive and non-zero
        adjusted_scores = [score + offset for score in fitness_scores]
    else:
        adjusted_scores = [score - min_fitness + 1 for score in fitness_scores]

    # Calculate selection probabilities using the Boltzmann formula
    exp_values = np.exp(np.array(adjusted_scores) / temperature)
    probabilities = exp_values / np.sum(exp_values)
    
    # Select new population based on calculated probabilities
    selected_indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
    new_population = [population[i] for i in selected_indices]

    return new_population


def tournament_selection(population, scored_population, tournament_rate=0.01):
    """Select individuals using tournament selection."""
    winners = []
    
    tournament_size = int(len(population) * tournament_rate)
    
    while len(winners) < len(population):
        tournament = random.sample(scored_population, tournament_size)
        tournament_winner = max(tournament, key=lambda x: x[1])
        winners.append(tournament_winner[0])
    return winners

def crossover(parent1, parent2):
    """Perform crossover between two parents to produce two offspring."""
    min_length = min(len(parent1), len(parent2))
    
    if min_length > 1:
        # Ensure crossover_point is chosen such that it's not the start or end of the sequences
        crossover_point = random.randint(1, min_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
    else:
        # For edge case where sequences are very short, direct cloning might be the fallback
        child1, child2 = parent1, parent2

    return child1, child2

def two_point_crossover(parent1, parent2):
    """Perform two-point crossover between two parents."""
    length = min(len(parent1), len(parent2))
    if length <= 2:
        return parent1, parent2  # Not enough length to crossover, return original parents
    
    # Generate two distinct crossover points
    cp1, cp2 = sorted(random.sample(range(1, length), 2))
    child1 = parent1[:cp1] + parent2[cp1:cp2] + parent1[cp2:]
    child2 = parent2[:cp1] + parent1[cp1:cp2] + parent2[cp2:]
    
    return child1, child2

def uniform_crossover(parent1, parent2):
    """Perform uniform crossover between two parents."""
    # Ensure both parents are of the same length for simplicity
    length = min(len(parent1), len(parent2))
    child1, child2 = [], []
    
    for i in range(length):
        # Randomly choose the parent to copy the gene from for each child
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    return child1, child2


if __name__ == "__main__":
    # Test the mutation functions
    
    index = 0
    for i in range(1000):
        individual_1 = [random.choice(list(Move)) for _ in range(20)]
        simplified = simplify_individual(individual_1)
        
        cube1 = Cube()
        cube1.move_list(individual_1)
        cube2 = Cube()
        cube2.move_list(simplified)
        
        if cube1.f == cube2.f:
            index += 1
        print(cube1.f == cube2.f)

    print(index)