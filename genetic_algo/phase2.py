from rubik54 import Cube
import random
from bayes_opt import BayesianOptimization
from utils.cube_utils import Move
from utils.ga_utils import boltzmann_selection, mutate, compute_fitness, \
                            crossover, generate_individual, load_model, \
                            simplify_individual, elitism, kill_by_rank, \
                            tournament_selection, uniform_crossover, \
                            two_point_crossover

# Main GA Loop
def genetic_algorithm_phase2(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate):
    population = [generate_individual(SEQUENCE_LENGTH, 2) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_individual = []
    index = 0
    sol_length = 0
        
    solved = False
    
    nnet = load_model()
    
    while index < NUM_GENERATIONS:
        # Score and select population
        scored_population = [(individual, compute_fitness(scrambled_str, individual, 2, nnet)) for individual in population]

        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)
        
        print(f"Generation {index} completed, Best Fitness: {best_fitness}")
        
        if best_fitness == 100:
            solved = True
            best_individual = simplify_individual(best_individual)
            best_individual = [move for move in best_individual if move != Move.N]
            sol_length = len(best_individual)
            break
        
        new_population = []        
        
        # Elitism: carry over top performers unchanged
        elite_individuals = elitism(scored_population, EliteRate)
                 
        # selected_population = kill_by_rank(scored_population, 0.5)
        selected_population = tournament_selection(population, scored_population, TournamentSize)
        # selected_population = boltzmann_selection(population, scored_population, temperature)

        elite_size = len(elite_individuals)

        while len(new_population) < (POPULATION_SIZE - elite_size):
            
            """ with prevention algorithm """
            child1, child2 = random.sample(selected_population, 2)
            # Mutate children and add them to the new population with simplification
            new_population.append(simplify_individual(mutate(child1, 2)))
            if len(new_population) < len(population):
                new_population.append(simplify_individual(mutate(child2, 2)))
            
            """ without prevention algorithm """
            # parent1, parent2 = random.sample(selected_population, 2)
            # child1, child2 = uniform_crossover(parent1, parent2)
            
            # # Mutate children and add them to the new population
            # new_population.append(mutate(child1, 1))
            # if len(new_population) < len(population):
            #     new_population.append(mutate(child2, 1))
        
        for i in elite_individuals:
            new_population.append(i)
            
        # Integrate elite individuals back into the population
        population = new_population
        
        index += 1
    
    return solved, index, sol_length, best_individual


def test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate):
    success = 0
    total_gen = 0
    total_len = 0
    
    for _ in range(100):
        # print("Iteration: ", i + 1)
        cube = Cube()
        cube.phase2_randomize_n(100)
        test_cube = cube.to_string()
        
        succeed, generations, sol_length, best_individual = genetic_algorithm_phase2(test_cube, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate)
        if succeed:
                success += 1
                total_gen += generations
                total_len += sol_length
                cube.move_list(best_individual)
                
    #     print(f"Success: {success}, Generations: {generations}, Solution Length: {sol_length}, Check Solved: {cube.is_solved()}")
    
    # print("Success rate: ", success / 100)
    # print("Average generations: ", total_gen / success)
    # print("Average solution length: ", total_len / success)

    # return the success rate
    return success - (total_len / success)

# test(3000, 1000, 23, 7, 0.007)


def function_to_be_optimized(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate):
    POPULATION_SIZE = int(POPULATION_SIZE)
    NUM_GENERATIONS = int(NUM_GENERATIONS)
    SEQUENCE_LENGTH = int(SEQUENCE_LENGTH)
    TournamentSize = int(TournamentSize)
    EliteRate = float(EliteRate)
    
    return test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate)

# Define the BayesianOptimization object
pbounds = {
    "POPULATION_SIZE": (1000, 5000),
    "NUM_GENERATIONS": (100, 300),
    "SEQUENCE_LENGTH": (10, 30),
    "TournamentSize": (2, 7),
    "EliteRate": (0.003, 0.009)
}

optimizer = BayesianOptimization(
    f=function_to_be_optimized,
    pbounds=pbounds,
    verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

optimizer.maximize(
    init_points=65,
    n_iter=35,
)

print(optimizer.max)