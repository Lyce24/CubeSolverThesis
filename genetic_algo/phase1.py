from rubik54 import Cube
import random
from utils.cube_utils import Move
# from bayes_opt import BayesianOptimization
from utils.ga_utils import boltzmann_selection, mutate, compute_fitness,generate_individual, simplify_individual, \
                            crossover, two_point_crossover, uniform_crossover

# Main GA Loop
def genetic_algorithm(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    population = [generate_individual(SEQUENCE_LENGTH, 1) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_individual = []
    index = 0
    sol_length = 0
    
    initial_temperature = TEMPERATURE
    temperature = initial_temperature
    cooling_rate = COOLING_RATE
    
    solved = False
    
    while index < NUM_GENERATIONS:
         # Score and select population
        scored_population = [(individual, compute_fitness(scrambled_str, individual, 1)) for individual in population]
        # scored_population = compute_fitness_population(scrambled_str, population)
        
        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)
                        
        if best_fitness == 100:
                solved = True
                best_individual = simplify_individual(best_individual)
                best_individual = [move for move in best_individual if move != Move.N]
                sol_length = len(best_individual)
                break
                
        selected_population = boltzmann_selection(population, scored_population, temperature)
        
        # Initialize new population with crossover and mutation
        new_population = []
        while len(new_population) < len(population):
            # Randomly select two parents for crossover
            child1, child2 = random.sample(selected_population, 2)
            #  = uniform_crossover(parent1, parent2)
            
            # Mutate children and add them to the new population
            new_population.append(simplify_individual(mutate(child1, 1)))
            if len(new_population) < len(population):
                new_population.append(simplify_individual(mutate(child2, 1)))

        population = new_population  # Update population with the newly generated one
        
        # Update temperature
        temperature *= cooling_rate
    
        index += 1
    
    return solved, index, sol_length, best_individual

def test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    success = 0
    total_gen = 0
    total_len = 0
    
    for i in range(100):
        print("Iteration: ", i + 1)
        cube = Cube()
        cube.randomize_n(100)
        test_cube = cube.to_string()
        
        succeed, generations, sol_length, best_individual = genetic_algorithm(test_cube, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE)
        if succeed:
                success += 1
                total_gen += generations
                total_len += sol_length
                cube.move_list(best_individual)
                
        print(f"Success: {success}, Generations: {generations}, Solution Length: {sol_length}, Check Slice: {cube.get_slice()}")
    
    print("Success rate: ", success / 100)
    print("Average generations: ", total_gen / success)
    print("Average solution length: ", total_len / success)

    # return the success rate
    return success / 100


def function_to_be_optimized(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    POPULATION_SIZE = int(POPULATION_SIZE)
    NUM_GENERATIONS = int(NUM_GENERATIONS)
    SEQUENCE_LENGTH = int(SEQUENCE_LENGTH)
    TEMPERATURE = float(TEMPERATURE)
    COOLING_RATE = float(COOLING_RATE)
    
    return test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE)

"""
    POPULATION_SIZE = 2402
    NUM_GENERATIONS = 249
    SEQUENCE_LENGTH = 25
    initial_temperature = 10.32
    cooling_rate = 0.989
""" 

test(2402, 249, 25, 10.32, 0.989)

# # Define the BayesianOptimization object
# pbounds = {
#     "POPULATION_SIZE": (1000, 5000),
#     "NUM_GENERATIONS": (100, 300),
#     "SEQUENCE_LENGTH": (10, 30),
#     "TEMPERATURE": (0, 100),
#     "COOLING_RATE": (0.9, 0.99)
# }

# optimizer = BayesianOptimization(
#     f=function_to_be_optimized,
#     pbounds=pbounds,
#     verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
#     random_state=1,
# )

# optimizer.maximize(
#     init_points=65,
#     n_iter=35,
# )

# print(optimizer.max)