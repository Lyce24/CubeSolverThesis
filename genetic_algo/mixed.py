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


def generate_individual():
    """Generate a random individual solution."""
    individual = [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]
    individual = simplify_individual(individual)
    return individual

def simplify_individual(individual):
    # Remove inverses and apply the new logic for simplification
    i = 0
    while i < len(individual):
        # Check for four same moves in a row
        if i < len(individual) - 3 and individual[i] == individual[i+1] == individual[i+2] == individual[i+3]:
            del individual[i:i+4]
            continue  # Continue checking from the current position

        # Check for three same moves in a row
        if i < len(individual) - 2 and individual[i] == individual[i+1] == individual[i+2]:
            # Replace three moves with their inverse
            individual[i] = inverse_moves[individual[i]]
            del individual[i+1:i+3]
            continue

        # Remove a move followed immediately by its inverse
        if i < len(individual) - 1 and individual[i+1] == inverse_moves.get(individual[i], ''):
            del individual[i:i+2]
            i = max(i - 1, 0)  # Step back to check for further simplifications
            continue
        i += 1

    return individual


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
    mutation_type = random.choice(['swap', 'scramble', 'invert', 'insert', 'delete'])
    
    if mutation_type == 'swap':
        # Swap two moves in the sequence
        if len(individual) > 1:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    
    elif mutation_type == 'scramble':
        # Scramble a subset of the sequence
        if len(individual) > 1:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end] = random.sample(individual[start:end], len(individual[start:end]))
    
    elif mutation_type == 'invert':
        # Invert a subset of the sequence
        if len(individual) > 1:
            start, end = sorted(random.sample(range(len(individual)), 2))
            individual[start:end] = individual[start:end][::-1]
    
    elif mutation_type == 'insert':
        # Insert a random move at a random position
        count = random.randint(1, 4)
        
        for _ in range(count):
            idx = random.randint(0, len(individual))
            move = random.choice(list(Move))
            individual.insert(idx, move)
    
    elif mutation_type == 'delete':
        # Delete a random move
        if len(individual) > 1:
            idx = random.randint(0, len(individual) - 1)
            del individual[idx]
    
    individual = simplify_individual(individual)
    
    return individual

def select_elites(scored_population, elite_size):
    # Sort the population by fitness in descending order
    sorted_population = sorted(scored_population, key=lambda x: x[1], reverse=True)
    # Select the top `elite_size` individuals
    elites = sorted_population[:elite_size]
    return [individual for individual, _ in elites]

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

def mixed_selection(population, scored_population, temperature, elite_rate = 0.1):
    elite_size = int(len(population) * elite_rate)
    # Select elites
    elites = select_elites(scored_population, elite_size)
    # Perform Boltzmann selection on the rest of the population
    non_elite_population = [individual for individual, _ in scored_population[elite_size:]]
    non_elite_scores = [score for _, score in scored_population[elite_size:]]
    if len(non_elite_population) > 0:  # Ensure there's a population to select from
        selected_non_elites = boltzmann_selection(non_elite_population, non_elite_scores, temperature)
    else:
        selected_non_elites = []
    # Combine elites with the selected non-elites
    new_population = elites + selected_non_elites
    return new_population


# Main GA Loop
def genetic_algorithm_with_plot(scrambled_str, model):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    # Training loop
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    best_fitnesses = []
    best_solution_found = ""
    index = 0
    
    initial_temperature = 100
    temperature = initial_temperature
    cooling_rate = 0.98
    min_temperature = 0.5
    
    nnet = load_model(model)
    
    while index < NUM_GENERATIONS:
         # Score and select population
        scored_population = [(individual, compute_fitness(scrambled_str, individual, model, nnet)) for individual in population]

        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)
        
        best_solution_found = ""
        for move in best_individual:
            best_solution_found += move_dict[move] + " "
        
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
                print("Best solution found:", best_solution_found)
                f.write("Best solution found: " + best_solution_found)
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
        temperature = max(temperature * cooling_rate, min_temperature)
    

        with open(output_file, 'a') as f:
            f.write(f"Generation {index + 1}: Best Fitness = {best_fitness}, Best Solution = {best_solution_found}, Temperature = {temperature}\n")
        print(f"Generation {index + 1}: Best Fitness = {best_fitness}, Best Solution = {best_solution_found}, Temperature = {temperature}")
        
        index += 1
    else:
        with open(output_file, 'a') as f:
            f.write("No solution found.\n")
            f.write("Best solution found: " + best_solution_found)
        print("No solution found.")
        print("Best solution found:", best_solution_found)
        
    # Save the plot
    plt.savefig(f"GA/mixed_fitnesses.png")
    plt.ioff()
    plt.show()
    
# Run the GA
if __name__ == "__main__":
    POPULATION_SIZE = 2000
    NUM_GENERATIONS = 1000
    MUTATION_RATE = 0.40
    SEQUENCE_LENGTH = 30
    output_file = "GA/output_mixed_drl.txt"
    
    with open(output_file, 'w') as f:
        f.write("Mixed Genetic Algorithm with DRL\n")
        test_cube = FaceCube()
        scramble_str = test_cube.randomize_n(18)
        print("Scrambled cube:", test_cube.to_2dstring())
        print("Scrambled cube:", scramble_str)
        print("Solving...")
        
        f.write(f"Scrambled cube: {scramble_str}\n")
        f.write("Scrambled cube: " + test_cube.to_2dstring() + "\n")
        f.write("Solving...\n")
        
        genetic_algorithm_with_plot(test_cube.to_string(), "resnet")