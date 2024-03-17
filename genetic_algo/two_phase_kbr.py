from rubik54 import Cube
import random
from utils.cube_utils import Move, phase2_move
import numpy as np
from bayes_opt import BayesianOptimization
from utils.mutate_utils import get_allowed_mutations, simplify_individual
from utils.fitness_utils import filter

def generate_individual(SEQUENCE_LENGTH):
    """Generate a random individual solution."""
    individual = [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]
    return (individual)

def compute_fitness(scrambled_str, individual):
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


def mutate(individual):
    # Randomly choose a gene (mutation point) to mutate
    gene_to_mutate = random.randint(0, len(individual) - 1)
    
    # Get the moves performed so far up to the mutation point
    move_sequence1 = individual[:gene_to_mutate]

    # get the moves performed after the mutation point
    if gene_to_mutate < len(individual) - 1:
        move_sequence2 = individual[gene_to_mutate + 1:]
    else:
        move_sequence2 = []
    
    # Get the list of allowed moves for the mutation based on the last move group
    allowed_moves = get_allowed_mutations(move_sequence1, move_sequence2)
        
    # Apply a valid mutation from the allowed moves
    individual[gene_to_mutate] = random.choice(allowed_moves)

    return (individual)


def kill_by_rank(scored_population, survival_rate=0.7):
    # Rank population by fitness
    scored_population.sort(key=lambda x: x[1], reverse=True)
    
    # Select survivors based on survival rate
    num_survivors = int(len(scored_population) * survival_rate)
    survivors = scored_population[:num_survivors]
    
    # Return survivors - first element of each tuple
    return [individual for individual, _ in survivors]

# Main GA Loop
def genetic_algorithm(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH):
    population = [generate_individual(SEQUENCE_LENGTH) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    index = 0
    sol_length = 0
        
    solved = False
    
    while index < NUM_GENERATIONS:
        # Score and select population
        scored_population = [(individual, compute_fitness(scrambled_str, individual)) for individual in population]

        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)
        
                
        if best_fitness == 20:
                solved = True
                best_individual = simplify_individual(best_individual)
                best_individual = [move for move in best_individual if move != Move.N]
                sol_length = len(best_individual)
                break
                
        selected_population = kill_by_rank(scored_population)
        
        # Initialize new population with crossover and mutation
        new_population = selected_population
        
        while len(new_population) < POPULATION_SIZE:
            # Randomly select two parents for crossover
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate children and add them to the new population
            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population  # Update population with the newly generated one
        
        index += 1
    else:
        pass
    
    return solved, index, sol_length

def test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH):
    success = 0
    total_gen = 0
    total_len = 0
    
    with open("scrambled.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            test_cube = line.split("\n")[0]
            print(f"Running test {i + 1}")
            
            succeed, generations, sol_length = genetic_algorithm(test_cube, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH)
            if succeed:
                success += 1
                total_gen += generations
                total_len += sol_length
        
            print(f"Success: {success}, Generations: {generations}, Solution Length: {sol_length}")
    
    print(f"Avg Generations: {total_gen / success}")
    print(f"Avg Solution Length: {total_len / success}")
    print(f"Success Rate: {success / 100}")
    
    # return the success rate
    return success / 100


test(8000, 1000, 23)