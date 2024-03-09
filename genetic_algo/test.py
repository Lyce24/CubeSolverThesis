import pygad
import numpy
from rubik54 import FaceCube, Move, move_dict
import random
import matplotlib.pyplot as plt
import torch
from utils.cube_model import MLPModel, ResnetModel
from collections import OrderedDict
import re
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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

# load model
def load_model(model_type):
    if model_type == "mlp":
        nnet = MLPModel(324).to(device)
        model = "saved_models/model102.dat"
        nnet.load_state_dict(torch.load(model, map_location=device))
        nnet.eval()
        return nnet
    
    elif model_type == "resnet":
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
    else:
        raise ValueError("Invalid model type")

nnet = load_model("resnet")
model = "resnet"

"""
Given a scrambled Rubik's Cube, the goal is to solve it using the genetic algorithm:
    - Each move is represented by a number from 0 to 11.
    - The fitness function is the number of steps required to solve the cube (simulated by a deep learning model).
    - The genetic algorithm will find the best sequence of moves to solve the cube.
    
This script uses the genetic algorithm to find the best sequence of moves to solve a scrambled Rubik's Cube.
"""
scramble_cube = FaceCube()
scramble_cube.randomize_n(18)
function_inputs = scramble_cube.to_string()
desired_output = 100 # Function output.


def fitness_func(ga_instance, solution, solution_idx):
    cube = FaceCube()
    cube.from_string(function_inputs)
    cube.move_list(solution)
    
    """Compute the fitness of an individual solution."""
    if cube.is_solved():   
        sol_str = (" ".join([move_dict[move] for move in solution]))
        print(f"Solution: {sol_str}")
        return 100
    else:
        if model == "mlp":
            input_state= cube.convert_mlp_input()
        elif model == "resnet":
            input_state = cube.convert_res_input()
        else:
            raise ValueError("Invalid model type")
        input_tensor = torch.tensor(input_state, dtype=torch.float32).to(device)
        output = nnet(input_tensor)
        if model == "mlp":
            return output.item()
        return -output.item()
    
def initialize_population():
    population = []
    for _ in range(sol_per_pop):
        individual = [int(random.choice(list(Move))) for _ in range(26)]
        population.append(individual)
    return population

fitness_function = fitness_func

num_generations = 500 # Number of generations.
num_parents_mating = 3000 # Number of solutions to be selected as parents in 
sol_per_pop = 3000 # Number of solutions in the population.
num_genes = 26 # Number of genes in the solution.

"sss (for steady-state selection), rws (for roulette wheel selection), sus (for stochastic universal selection), rank (for rank selection), random (for random selection), and tournament (for tournament selection)."
parent_selection_type = "rank"

"single_point (for single-point crossover), two_points (for two points crossover), uniform (for uniform crossover), and scattered (for scattered crossover)."
crossover_type = "uniform"

"random (for random mutation), swap (for swap mutation), inversion (for inversion mutation), scramble (for scramble mutation), and adaptive (for adaptive mutation)."
mutation_type = "random"


keep_parents = 20
keep_elitism = 10

mutation_percent_genes = 95
crossover_probability=None



last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")
    print(f"Change     = {ga_instance.best_solution()[1] - last_fitness}\n")
    last_fitness = ga_instance.best_solution()[1]


# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       on_generation=callback_generation,
                       initial_population=initialize_population(),
                       gene_type=int,
                       stop_criteria="reach_100",
                       parent_selection_type = parent_selection_type,
                       keep_parents = keep_parents,
                       keep_elitism = keep_elitism,
                       crossover_type=crossover_type,
                       mutation_type = mutation_type,
                       mutation_probability = mutation_percent_genes/100)
                       
                       
# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_fitness()

# Returning the details of the best solution.
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")
print(f"Index of the best solution : {solution_idx}")

prediction = numpy.sum(numpy.array(function_inputs)*solution)
print(f"Predicted output based on the best solution : {prediction}")

if ga_instance.best_solution_generation != -1:
    print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = pygad.load(filename=filename)
loaded_ga_instance.plot_fitness()