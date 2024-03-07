from rubik54 import FaceCube, Move, move_dict
import random
import matplotlib.pyplot as plt
import torch
from utils.cube_model import MLPModel, ResnetModel
from collections import OrderedDict
import re

# Genetic Algorithm Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

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
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)
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


def generate_individual():
    """Generate a random individual solution."""
    return [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]

def compute_fitness(scrambled_str, individual, model, nnet):
    cube = FaceCube()
    cube.from_string(scrambled_str)
    cube.move_list(individual)
    
    """Compute the fitness of an individual solution."""
    if cube.is_solved():
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
    
def generate_children(parents):
    """Generate two children from two parents."""
    contest_1 = random.sample(parents, 2)
    winner_1 = contest_1[0] if contest_1[0][1] > contest_1[1][1] else contest_1[1]
    loser_1 = contest_1[0] if winner_1 == contest_1[1] else contest_1[1]
    contest_2 = random.sample(parents, 2)
    winner_2 = contest_2[0] if contest_2[0][1] > contest_2[1][1] else contest_2[1]
    loser_2 = contest_2[0] if winner_2 == contest_2[1] else contest_2[1]
    
    child1, child2 = crossover(winner_1[0], winner_2[0])
    child3, child4 = crossover(loser_1[0], loser_2[0])
    
    return child1, child2, child3, child4

def crossover(parent1, parent2):
    """Perform crossover between two parents to produce two offspring."""
    crossover_point = random.randint(0, SEQUENCE_LENGTH - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual):
    """Mutate an individual's genes."""
    for i in range(SEQUENCE_LENGTH):
        if random.random() < MUTATION_RATE:
            individual[i] = random.choice(list(Move))
    return individual

def kill_by_rank(population, scrambled_str, model, nnet, survival_rate=0.5):
    # Calculate fitness for each individual
    scored_population = [(individual, compute_fitness(scrambled_str, individual, model, nnet)) for individual in population]
    
    # Rank population by fitness
    scored_population.sort(key=lambda x: x[1], reverse=True)

    # Find the maximum fitness and best responce in the population for logging purposes
    best_fitness = scored_population[0][1]
    best_individual = scored_population[0][0]
    
    # Select survivors based on survival rate
    num_survivors = int(len(population) * survival_rate)
    survivors = scored_population[:num_survivors]
    
    # Generate offspring
    offspring = []
    while len(offspring) < (len(population) - num_survivors):
        
        child1, child2, child3, child4 = generate_children(survivors)
        offspring.append(mutate(child1))
        if len(offspring) < (len(population) - num_survivors):
            offspring.append(mutate(child2))
        
        if len(offspring) < (len(population) - num_survivors):
            offspring.append(mutate(child3))
            
        if len(offspring) < (len(population) - num_survivors):
            offspring.append(mutate(child4))
            
    # Create new population by combining survivors and offspring
    new_population = [individual for individual, _ in survivors] + offspring
    return new_population, best_fitness, best_individual

# Main GA Loop
def genetic_algorithm_with_plot(scrambled_str, model):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    # Training loop
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    best_fitnesses = []
    best_solution_found = ""
    index = 0
    
    nnet = load_model(model)
    
    while index < NUM_GENERATIONS:
        population, best_fitness, best_individual = kill_by_rank(population, scrambled_str, model, nnet)
    
        best_fitnesses.append(best_fitness)
        
        ax.clear()
        ax.plot(best_fitnesses, label='Training Best Fitnesses')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Best Fitnesses Over Generations')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

        best_solution_found = ""
        for move in best_individual:
            best_solution_found += move_dict[move] + " "
                
        if best_fitness == 100:
            with open(output_file, 'a') as f:
                f.write("Solution found!\n")
                print("Solution found!")
                print("Best solution found:", best_solution_found)
                f.write("Best solution found: " + best_solution_found)
                break
            
        with open(output_file, 'a') as f:
            f.write(f"Generation {index + 1}: Best Fitness = {best_fitness}\n")
        print(f"Generation {index + 1}: Best Fitness = {best_fitness}")
        index += 1
    else:
        with open(output_file, 'a') as f:
            f.write("No solution found.\n")
            f.write("Best solution found: " + best_solution_found)
        print("No solution found.")
        print("Best solution found:", best_solution_found)
        
    # Save the plot
    plt.savefig(f"GA/DRL_loss.png")
    plt.ioff()
    plt.show()

# # Main GA Loop without plot
# def genetic_algorithm(scrambled_str, model):
#     population = [generate_individual() for _ in range(POPULATION_SIZE)]
#     # Training loop
    
#     index = 0
#     while index < NUM_GENERATIONS:
#         population, best_fitness = kill_by_rank(population, scrambled_str, model)
            
#         if best_fitness == 100:
#             with open(output_file, 'a') as f:
#                 f.write("Solution found!\n")
#                 print("Solution found!")
#                 best_individual = max(population, key=lambda individual: compute_fitness(scrambled_str, individual, model))
#                 best_str = ""
#                 for move in best_individual:
#                     best_str += move_dict[move] + " "
                    
#                 print("Best solution:", best_str)
#                 f.write(best_str)
#                 return 1
            
#         with open(output_file, 'a') as f:
#             f.write(f"Generation {index + 1}: Best Fitness = {best_fitness}\n")
#         print(f"Generation {index + 1}: Best Fitness = {best_fitness}")
#         index += 1
#     else:
#         with open(output_file, 'a') as f:
#             f.write("No solution found.\n")
#         print("No solution found.")
#         return 0

    
# Run the GA
if __name__ == "__main__":
    POPULATION_SIZE = 3000
    NUM_GENERATIONS = 500
    MUTATION_RATE = 0.40
    SEQUENCE_LENGTH = 20
    output_file = "GA/output_drl.txt"
    
    test_cube = FaceCube()
    test_cube.randomize_n(25)
    print("Scrambled cube:", test_cube.to_2dstring())
    print("Scrambled cube:", test_cube.to_string())
    print("Solving...")
    genetic_algorithm_with_plot(test_cube.to_string(), "resnet")