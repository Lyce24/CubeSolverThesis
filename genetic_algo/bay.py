from rubik54 import Cube
import random
from utils.cube_utils import Move, move_dict
import numpy as np
from bayes_opt import BayesianOptimization
from utils.mutate_utils import get_allowed_mutations


def generate_individual(SEQUENCE_LENGTH):
    """Generate a random individual solution."""
    individual = [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]
    # individual = simplify_individual(individual)
    return individual


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

    return individual


def boltzmann_selection(population, scored_population, temperature):
    # Extract fitness scores and normalize them
    fitness_scores = [score for _, score in scored_population]
    max_fitness = max(fitness_scores)
    normalized_fitness_scores = [score - max_fitness for score in fitness_scores]  # Normalize scores to avoid overflow

    # Calculate selection probabilities using the Boltzmann formula
    exp_values = np.exp(np.array(normalized_fitness_scores) / temperature)
    probabilities = exp_values / np.sum(exp_values)

    # Select new population based on calculated probabilities
    selected_indices = np.random.choice(len(population), size=len(population), replace=True, p=probabilities)
    new_population = [population[i] for i in selected_indices]

    return new_population

# Main GA Loop
def genetic_algorithm(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    population = [generate_individual(SEQUENCE_LENGTH) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_solution_found = ""
    index = 0
    
    initial_temperature = TEMPERATURE
    temperature = initial_temperature
    cooling_rate = COOLING_RATE
    
    solved = False
    
    while index < NUM_GENERATIONS:
         # Score and select population
        scored_population = [(individual, compute_fitness_phase1(scrambled_str, individual)) for individual in population]

        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)
        
        best_solution_found = ""
        for move in best_individual:
            best_solution_found += move_dict[move] + " "
                
        if best_fitness == 20:
                solved = True
                break
                
        selected_population = boltzmann_selection(population, scored_population, temperature)
        
        # Initialize new population with crossover and mutation
        new_population = []
        while len(new_population) < len(population):
            # Randomly select two parents for crossover
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate children and add them to the new population
            new_population.append(mutate(child1))
            if len(new_population) < len(population):
                new_population.append(mutate(child2))

        population = new_population  # Update population with the newly generated one
        
        # Update temperature
        temperature *= cooling_rate
    
        index += 1
    else:
        pass
    
    return solved, index

def test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    success = 0
    total_gen = 0
    for i in range(100):
        print(f"Running test {i + 1}")
        test_cube = Cube()
        test_cube.randomize_n(100)
        
        succeed, generations =  genetic_algorithm(test_cube.to_string(), POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE)
        if succeed:
            success += 1
            total_gen += generations
        
        print(f"Success: {success}, Generations: {generations}")
    
    print(f"Avg Generations: {total_gen / success}")
    # return the success rate
    return success / 100

"""
    POPULATION_SIZE = 4205
    NUM_GENERATIONS = 270
    MUTATION_RATE = 0.40
    SEQUENCE_LENGTH = 23
"""

def function_to_be_optimized(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    POPULATION_SIZE = int(POPULATION_SIZE)
    NUM_GENERATIONS = int(NUM_GENERATIONS)
    SEQUENCE_LENGTH = int(SEQUENCE_LENGTH)
    TEMPERATURE = float(TEMPERATURE)
    COOLING_RATE = float(COOLING_RATE)
    
    return test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE)

# Define the BayesianOptimization object
pbounds = {
    "POPULATION_SIZE": (1000, 6000),
    "NUM_GENERATIONS": (100, 400),
    "SEQUENCE_LENGTH": (10, 30),
    "TEMPERATURE": (0, 100),
    "COOLING_RATE": (0.9, 0.99)
}

optimizer = BayesianOptimization(
    f=function_to_be_optimized,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=10,
    n_iter=10,
)

print(optimizer.max)