from rubik54 import FaceCube, Move, move_dict, color_dict
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from utils.cube_model import MLPModel

# Genetic Algorithm Parameters
POPULATION_SIZE = 2000
NUM_GENERATIONS = 500
MUTATION_RATE = 0.33
SEQUENCE_LENGTH = 10
output_file = "GA/output_drl.txt"

# load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
net = MLPModel(324).to(device)
model = "saved_models/model102.dat"
net.load_state_dict(torch.load(model, map_location=device))
net.eval()

def generate_individual():
    """Generate a random individual solution."""
    return [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]

def compute_fitness(scrambled_str, individual):
    cube = FaceCube()
    cube.from_string(scrambled_str)
    cube.move_list(individual)
    
    """Compute the fitness of an individual solution."""
    if cube.is_solved():
        return 100
    else:
        input_tensor = torch.tensor(cube.convert_nnet_input(), dtype=torch.float32).to(device)
        output = net(input_tensor)
        return output.item()

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

def kill_by_rank(population, scrambled_str, survival_rate=0.7):
    # Calculate fitness for each individual
    scored_population = [(individual, compute_fitness(scrambled_str, individual)) for individual in population]
    
    # Rank population by fitness
    scored_population.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness in descending order
    
    # Find the maximum fitness in the population for logging purposes
    max_fitness = scored_population[0][1]
    
    # Select survivors based on survival rate
    num_survivors = int(len(population) * survival_rate)
    survivors = scored_population[:num_survivors]
    
    # Generate offspring
    offspring = []
    while len(offspring) < (len(population) - num_survivors):
        parent1, parent2 = random.sample([individual for individual, _ in survivors], 2)
       
        child1, child2 = crossover(parent1, parent2)
        offspring.append(mutate(child1))
        if len(offspring) < (len(population) - num_survivors):
            offspring.append(mutate(child2))
    
    # Create new population by combining survivors and offspring
    new_population = [individual for individual, _ in survivors] + offspring
    return new_population, max_fitness

# Main GA Loop
def genetic_algorithm_with_plot(scrambled_str):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    # Training loop
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    best_fitnesses = []
    
    index = 0
    while index < NUM_GENERATIONS:
        population, best_fitness = kill_by_rank(population, scrambled_str)
    
        best_fitnesses.append(best_fitness)
        
        ax.clear()
        ax.plot(best_fitnesses, label='Training Best Fitnesses')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.set_title('Best Fitnesses Over Generations')
        ax.legend()
        fig.canvas.draw()
        fig.canvas.flush_events()

            
        if best_fitness == 100:
            with open(output_file, 'a') as f:
                f.write("Solution found!\n")
                print("Solution found!")
                best_individual = max(population, key=lambda individual: compute_fitness(scrambled_str, individual))
                best_str = ""
                for move in best_individual:
                    best_str += move_dict[move] + " "
                    
                print("Best solution:", best_str)
                f.write(best_str)
                break
            
        with open(output_file, 'a') as f:
            f.write(f"Generation {index + 1}: Best Fitness = {best_fitness}\n")
        print(f"Generation {index + 1}: Best Fitness = {best_fitness}")
        index += 1
    else:
        with open(output_file, 'a') as f:
            f.write("No solution found.\n")
        print("No solution found.")
        
    # Save the plot
    plt.savefig(f"GA/DRL_loss.png")
    plt.ioff()
    plt.show()

# Main GA Loop without plot
def genetic_algorithm(scrambled_str):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    # Training loop
    
    index = 0
    while index < NUM_GENERATIONS:
        population, best_fitness = kill_by_rank(population, scrambled_str)
            
        if best_fitness == 100:
            with open(output_file, 'a') as f:
                f.write("Solution found!\n")
                print("Solution found!")
                best_individual = max(population, key=lambda individual: compute_fitness(scrambled_str, individual))
                best_str = ""
                for move in best_individual:
                    best_str += move_dict[move] + " "
                    
                print("Best solution:", best_str)
                f.write(best_str)
                return 1
            
        with open(output_file, 'a') as f:
            f.write(f"Generation {index + 1}: Best Fitness = {best_fitness}\n")
        print(f"Generation {index + 1}: Best Fitness = {best_fitness}")
        index += 1
    else:
        with open(output_file, 'a') as f:
            f.write("No solution found.\n")
        print("No solution found.")
        return 0

    
# Run the GA
if __name__ == "__main__":
    test_cube = FaceCube()
    test_cube.move_list(test_cube.convert_move("L2 R U2 B' B' L' R' R2 U' U L F' R L2 U L' L' R D' D2"))
    print("Scrambled cube:", test_cube.to_2dstring())
    print("Scrambled cube:", test_cube.to_string())
    print("Solving...")
    genetic_algorithm_with_plot(test_cube.to_string())