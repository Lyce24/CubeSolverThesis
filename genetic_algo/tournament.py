from rubik54 import FaceCube, Move, move_dict
import random
import matplotlib.pyplot as plt
import torch
from utils.cube_model import MLPModel, ResnetModel
from collections import OrderedDict
import re


output_file = 'drl_tournament.txt'


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


# Genetic Algorithm Parameters
POPULATION_SIZE = 1000
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.8
TOURNAMENT_SIZE = 200
SEQUENCE_LENGTH = 20 # Initial sequence length for each individual


def generate_individual():
    """Generate a random individual solution."""
    return [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]

def compute_fitness(scrambled_str, individual):
    """Compute the fitness of an individual solution."""
    cube = FaceCube()
    cube.from_string(scrambled_str)
    # Initialize a solved cube
    solved_cube = FaceCube()
    # Apply the sequence of moves from the individual to a new cube
    cube.move_list(individual)
    
    # Compare each subface to the solved state
    correct_subfaces = sum(1 for i in range(54) if cube.f[i] == solved_cube.f[i])
    return correct_subfaces

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


def tournament_selection(population, scores):
    """Select one individual using tournament selection."""
    tournament = random.sample(list(zip(population, scores)), TOURNAMENT_SIZE)
    tournament.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness in descending order
    return tournament[0][0]

def genetic_algorithm_tournament(scrambled_str, model, nnet):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    max_fitness_over_generations = []
    
    for generation in range(NUM_GENERATIONS):
        population, max_fitness, best_individual = kill_by_rank(population, scrambled_str, model, nnet)
        max_fitness_over_generations.append(max_fitness)
        
        if max_fitness == 100:
            print("Solution found!")
            print("Best individual:", best_individual)
            break
    else:
        print("No solution found.")
    
    return max_fitness_over_generations

scrambled_cube = FaceCube()
with open(output_file, 'w') as f:
    f.write(scrambled_cube.randomize())
    f.write('\n')
    f.write(scrambled_cube.to_2dstring())

print("Scrambled cube:")
print(scrambled_cube.to_2dstring())
scrambled_str = scrambled_cube.to_string()

# Initialize population and fitness tracking
population = [generate_individual() for _ in range(POPULATION_SIZE)]
max_fitness_over_generations = []

# Initialize population
population = [generate_individual() for _ in range(POPULATION_SIZE)]

# Evolution loop
for generation in range(NUM_GENERATIONS):
    start_time = time.time()
    # Compute fitness for each individual
    scores = compute_fitnesses(population)
    max_fitness = max(scores)
    max_fitness_over_generations.append(max_fitness)

    # Check for a solution
    if max(scores) == 54:  # A score of 54 indicates a solved cube
        with open(output_file, 'a') as f:
            f.write("Solution found!")
            print("Solution found!")
            best_index = scores.index(max(scores))
            best_solution = population[best_index]
            best_str = ""
            for move in best_solution:
                best_str += move_dict[move] + " "
                
            print("Best solution:", best_str)
            f.write(best_str)
            break

    # Selection and reproduction
    new_population = []
    while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            child1, child2 = crossover(parent1, parent2)
            new_population.extend([mutate(child1), mutate(child2)])

    population = new_population[:POPULATION_SIZE]
    
    finish_time = time.time()
    completed_time = finish_time - start_time
    
    with open(output_file, 'a') as f:
        f.write(f"Generation {generation}: Max Fitness = {max(scores)}, Time = {completed_time}")
        print(f"Generation {generation}: Max Fitness = {max(scores)}, Time = {completed_time}")

else:
    print("No solution found.")