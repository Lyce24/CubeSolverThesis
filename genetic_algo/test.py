from rubik54 import Cube
import random
from bayes_opt import BayesianOptimization
from utils.cube_utils import Move, move_dict
from utils.test_utils import boltzmann_selection, mutate, compute_fitness, \
                            crossover, generate_individual, \
                            elitism, kill_by_rank, \
                            tournament_selection, uniform_crossover, \
                            two_point_crossover
                            
import torch
from utils.cube_model import ResnetModel
from collections import OrderedDict
import re


# Genetic Algorithm Parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Main GA Loop
def genetic_algorithm(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate):
    population = [generate_individual(SEQUENCE_LENGTH, 1) for _ in range(POPULATION_SIZE)]
    
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
            best_individual = (best_individual)
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
            # child1, child2 = random.sample(selected_population, 2)
            # # Mutate children and add them to the new population with simplification
            # new_population.append(simplify_individual(mutate(child1, 1)))
            # if len(new_population) < len(population):
            #     new_population.append(simplify_individual(mutate(child2, 1)))
            
            """ without prevention algorithm """
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = uniform_crossover(parent1, parent2)
            
            # Mutate children and add them to the new population
            new_population.append((mutate(child1, 1)))
            if len(new_population) < len(population):
                new_population.append((mutate(child2, 1)))
        
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
    
    for i in range(100):
        print("Iteration: ", i + 1)
        cube = Cube()
        print(cube.randomize_n(100))
        test_cube = cube.to_string()
        
        succeed, generations, sol_length, best_individual = genetic_algorithm(test_cube, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate)
        if succeed:
                success += 1
                total_gen += generations
                total_len += sol_length
                cube.move_list(best_individual)
        
        best_sol = ""
        for move in best_individual:
            if move != Move.N:
                best_sol += move_dict[move] + " "
                
        print(f"Best Solution: {best_sol}")
        print(f"Success: {success}, Generations: {generations}, Solution Length: {sol_length}, Check Solved: {cube.is_solved()}")
    
    print("Success rate: ", success / 100)
    print("Average generations: ", total_gen / success)
    print("Average solution length: ", total_len / success)

    # return the success rate
    return success - (total_len / success)

if __name__ == "__main__":
    test(8000, 1000, 100, 3, 0.005)


# def function_to_be_optimized(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate):
#     POPULATION_SIZE = int(POPULATION_SIZE)
#     NUM_GENERATIONS = int(NUM_GENERATIONS)
#     SEQUENCE_LENGTH = int(SEQUENCE_LENGTH)
#     TournamentSize = int(TournamentSize)
#     EliteRate = float(EliteRate)
    
#     return test(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate)

# # Define the BayesianOptimization object
# pbounds = {
#     "POPULATION_SIZE": (1000, 5000),
#     "NUM_GENERATIONS": (100, 300),
#     "SEQUENCE_LENGTH": (10, 30),
#     "TournamentSize": (2, 7),
#     "EliteRate": (0.003, 0.009)
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