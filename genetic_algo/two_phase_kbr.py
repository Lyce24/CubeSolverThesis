from rubik54 import Cube
import random
import matplotlib.pyplot as plt
import torch
from utils.cube_model import MLPModel, ResnetModel
from collections import OrderedDict
import re
from utils.mutate_utils import get_allowed_mutations
import time
from utils.selection_utils import phase2_filters
from utils.cube_utils import Facelet, Color, Corner, Edge, Move, BS, cornerFacelet, edgeFacelet, cornerColor, edgeColor, move_dict, color_dict


# Genetic Algorithm Parameters

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
    cube = Cube()
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
    
def compute_fitness_phase1(scrambled_str, individual, model, nnet):
    """Compute the fitness of an individual solution."""
        
    filters = phase2_filters(individual)
        
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
            
    return total/filters
    
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

# def mutate(individual):
#     # Single point mutation
#     mutation_point = random.randint(0, len(individual) - 1)
#     individual[mutation_point] = random.choice(list(Move))

#     return individual


def mutate(individual):
    # Randomly choose a gene (mutation point) to mutate
    gene_to_mutate = random.randint(0, SEQUENCE_LENGTH - 1)
    
    # Get the moves performed so far up to the mutation point
    move_sequence_up_to_point = individual[:gene_to_mutate]
    
    # Get the list of allowed moves for the mutation based on the last move group
    allowed_moves = get_allowed_mutations(move_sequence_up_to_point)
 
    # Apply a valid mutation from the allowed moves
    individual[gene_to_mutate] = random.choice(allowed_moves)

    return individual


def kill_by_rank(population, scrambled_str, model, nnet, survival_rate=0.7):
    # Calculate fitness for each individual
    scored_population = [(individual, compute_fitness_phase1(scrambled_str, individual, model, nnet)) for individual in population]
    
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

def simplify_individual(individual):
    # Remove inverses and apply the new logic for simplification
    i = 0
    
    temp = []
    for i in individual:
        if i != Move.N:
            temp.append(i)
    
    individual = temp    
    
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

# Main GA Loop
def genetic_algorithm(scrambled_str, model, mode = "normal", verbose = False):
    population = [generate_individual() for _ in range(POPULATION_SIZE)]
    # Training loop
    
    if mode == "plot":
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 6))
        
    best_fitnesses = []    
    best_solution_found = ""
    index = 0
    nnet = load_model(model)
    success = False
    
    while index < NUM_GENERATIONS:
        population, best_fitness, best_individual = kill_by_rank(population, scrambled_str, model, nnet)
    
        best_fitnesses.append(best_fitness)
        
        if mode == "plot":
            ax.clear()
            ax.plot(best_fitnesses, label='Training Best Fitnesses')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.set_title('Best Fitnesses Over Generations')
            ax.legend()
            fig.canvas.draw()
            fig.canvas.flush_events()

        best_solution_found = ""
        for move in simplify_individual(best_individual):
            best_solution_found += move_dict[move] + " "
                
        if best_fitness == 20:
            with open(output_file, 'a') as f:
                f.write("Solution found!\n")
                print("Solution found!")
                print(f"Best solution found: {best_solution_found} in {index + 1} generations.\n")
                f.write(f"Best solution found: {best_solution_found} in {index + 1} generations.\n")
                success = True
                break
        if verbose:
            print(f"Generation {index + 1}: Best Fitness = {best_fitness}, Best Solution = {best_solution_found}")
        index += 1
    else:
        with open(output_file, 'a') as f:
            f.write("No solution found.\n")
            f.write(f"Best solution found: {best_solution_found}.\n")
        print("No solution found.")
       
    if mode == "plot": 
        # Save the plot
        plt.savefig(f"GA/kbr_drl_loss.png")
        plt.ioff()
        plt.show()
    
    return success, best_fitness, best_solution_found, index
    
    
def test_100(scramble_length):
    success = 0
    generations = 0
    for i in range(100):
        print(f"Test {i+1}")
            
        start_time = time.time()
        scramble_cube = Cube()
       
        scramble_str = scramble_cube.randomize_n(scramble_length)
            
        print("Scrambled cube:", scramble_str)
        print("Scrambled cube:\n", scramble_cube.to_2dstring())
        print("Solving...")
        
        succeed, _, _, generation = genetic_algorithm(scramble_cube.to_string(), "resnet")
        if succeed:
            success += 1
            generations += generation
            print(f"Success! Generations: {generation}")
        
        if success != 0:
            print(f"Average generations: {generations/success}")
            
        print(f"Time taken: {time.time() - start_time} seconds")
        # calculate how much time is left
        print(f"Time left: {(time.time() - start_time) * (100 - i - 1)} seconds")
        
    
    print(f"Success rate: {success}, Average generations: {generations/success}")
    with open(output_file, 'a') as f:
        f.write(f"Success rate: {success}, Average generations: {generations/success}\n")

# Run the GA
if __name__ == "__main__":
    POPULATION_SIZE = 1000
    NUM_GENERATIONS = 1000
    SEQUENCE_LENGTH = 11
    output_file = "GA/two_phase/test.txt"
    
    # # create output file if not exists
    # with open(output_file, 'w') as f:
    #     f.write("Scramble length: 15\n")
    #     f.write("Population size: 3000\n")
    #     f.write("Number of generations: 1000\n")
    #     f.write("Sequence length: 20\n")
    
    # test_100(15)
    with open(output_file, 'w') as f:
        
        scramble_cube = Cube()
        scramble_str = scramble_cube.randomize_n(100)
        
        f.write(f"Scrambled cube: {scramble_str}\n")
        f.write("Scrambled cube:\n" + scramble_cube.to_2dstring() + "\n")
        f.write("Solving...\n")
        
        print("Scrambled cube:", scramble_str)
        print("Scrambled cube:\n", scramble_cube.to_2dstring())
        print("Solving...")
        
        genetic_algorithm(scramble_cube.to_string(), "resnet", "plot", verbose=True)