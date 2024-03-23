from rubik54 import Cube
import random
from bayes_opt import BayesianOptimization
from cube_utils import Move, move_dict
from search_utils import mutate, \
                            crossover, generate_individual, \
                            elitism, kill_by_rank, \
                            tournament_selection, uniform_crossover, \
                            two_point_crossover
                            
import torch
from ga_utils import ResnetModel
from collections import OrderedDict
import re
import numpy as np


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
def genetic_algorithm(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate, threshold):
    population = [generate_individual(SEQUENCE_LENGTH, 1) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_individual = []
    index = 0
    sol_length = 0
        
    solved = False
    
    nnet = load_model()
    
    while index < NUM_GENERATIONS:
        # Score and select population
        batch_states = []
        batch_info = []  # To keep track of the corresponding cube and moves
        
        for individual in population:
            cube = Cube()
            cube.from_string(scrambled_str)
            cube.move_list(individual)
            if cube.is_solved():
                solved = True
                best_individual = individual
                sol_length = len(best_individual)
                break
            
            batch_states.append(cube.convert_res_input())
            batch_info.append((cube, individual))
            
        states_np = np.array(batch_states)
        # Convert NumPy array to a PyTorch tensor
        input_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
        # Compute output in a single batch
        outputs = nnet(input_tensor)
        fitness_score = outputs.detach().cpu().numpy()
        
        scored_population = []
         # Create Beam_Node instances using the fitness scores
        for (_, individual), fitness in zip(batch_info, fitness_score):
            scored_population.append((individual, -fitness))
            

        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)

        
        print(f"Generation {index} completed, Best Fitness: {best_fitness}")
        
        if best_fitness > threshold:
            solved = True
            best_individual = (best_individual)
            best_individual = [move for move in best_individual if move != Move.N]
            sol_length = len(best_individual)
            break
        
        new_population = []        
        
        # Elitism: carry over top performers unchanged
        elite_individuals = elitism(scored_population, EliteRate)
                 
        # selected_population = kill_by_rank(scored_population, 0.7)
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
            new_population.append(mutate(child1, 1))
            if len(new_population) < len(population):
                new_population.append(mutate(child2, 1))
        
        for i in elite_individuals:
            new_population.append(i)
            
        # Integrate elite individuals back into the population
        population = new_population
        
        index += 1
    
    return solved, index, sol_length, best_individual


if __name__ == "__main__":
    test_str = "B U' R U U R U D R F R B R' F' U' R' F' F' U R' F' D"

    cube = Cube()
    cube.move_list(cube.convert_move(test_str))
    
    # [<Move.R3: 5>, <Move.F1: 6>, <Move.L3: 14>, <Move.U1: 0>, <Move.R1: 3>, <Move.R1: 3>, <Move.D1: 9>, <Move.F3: 8>, <Move.R3: 5>, <Move.U1: 0>, <Move.F3: 8>, <Move.L3: 14>, <Move.B3: 17>, <Move.D3: 11>, <Move.B3: 17>, <Move.D1: 9>, <Move.R1: 3>, <Move.D3: 11>, <Move.B1: 15>, <Move.D1: 9>, <Move.B1: 15>, <Move.U3: 2>]
    sol = [Move.R3, Move.F1, Move.L3, Move.U1, Move.R1, Move.R1, Move.D1, Move.F3, Move.R3, Move.U1, Move.F3, Move.L3, Move.B3, Move.D3, Move.B3, Move.D1, Move.R1, Move.D3, Move.B1, Move.D1, Move.B1, Move.U3]

    print(cube.is_solved())

    # # iteration 1
    # succeed, generations, sol_length, best_individual = genetic_algorithm(cube.to_string(), 8000, 1000, 26, 5, 0.004, threshold=99)
    