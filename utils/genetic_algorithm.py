from cube import Cube, load_model, Move, device, get_allowed_moves
import random                            
import torch
import numpy as np

nnet = load_model()

def generate_individual(SEQUENCE_LENGTH):
    """Generate a random individual solution."""
    individual = [random.choice(list(Move)) for _ in range(SEQUENCE_LENGTH)]
    return individual

def compute_fitness(states):
    # Convert list of states to a NumPy array (if not already an array)
    states_np = np.array(states)
    # Convert NumPy array to a PyTorch tensor
    input_tensor = torch.tensor(states_np, dtype=torch.float32).to(device)
    # Compute output in a single batch
    outputs = nnet(input_tensor)
    fitness_scores = outputs.detach().cpu().numpy()
    torch.cuda.empty_cache()  # Clear CUDA cache here
    return fitness_scores

def elitism(scored_population, elite_rate = 0.007):
    """Carry over the top individuals to the next generation."""
    scored_population.sort(key=lambda x: x[1], reverse=True)
    elite_size = int(len(scored_population) * elite_rate)
    elite_individuals = [individual for individual, _ in scored_population[:elite_size]]
    return elite_individuals

def tournament_selection(population, scored_population, tournament_size = 2):
    """Select individuals using tournament selection."""
    winners = []
    
    if tournament_size < 2:
        raise ValueError("Tournament size must be at least 2")
    
    while len(winners) < len(population):
        tournament = random.sample(scored_population, tournament_size)
        tournament_winner = max(tournament, key=lambda x: x[1])
        winners.append(tournament_winner[0])
    return winners

def kill_by_rank(scored_population, survival_rate=0.7):
    # Rank population by fitness
    scored_population.sort(key=lambda x: x[1], reverse=True)
    
    # Select survivors based on survival rate
    num_survivors = int(len(scored_population) * survival_rate)
    survivors = scored_population[:num_survivors]
    
    # Return survivors - first element of each tuple
    return [individual for individual, _ in survivors]

def uniform_crossover(parent1, parent2):
    """Perform uniform crossover between two parents."""
    # Ensure both parents are of the same length for simplicity
    length = min(len(parent1), len(parent2))
    child1, child2 = [], []
    
    for i in range(length):
        # Randomly choose the parent to copy the gene from for each child
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])
    
    return child1, child2


def mutate(individual, mutation_rate=None):
    if mutation_rate:
        # Randomly choose whether to mutate
        if random.random() > mutation_rate:
            return individual
    
    # Randomly choose a gene (mutation point) to mutate
    gene_to_mutate = random.randint(0, len(individual) - 1)
    
    # Get the moves performed so far up to the mutation point
    move_sequence = individual[:gene_to_mutate]
    
    allowed_moves = []

    allowed_moves = get_allowed_moves(move_sequence)
    
    # Apply a valid mutation from the allowed moves
    new_individual = individual.copy()
    new_individual[gene_to_mutate] = random.choice(allowed_moves)

    return new_individual
    
# Main GA Loop
def genetic_algorithm(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TournamentSize, EliteRate):
    population = [generate_individual(SEQUENCE_LENGTH) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_individual = []
    index = 0
    sol_length = 0
        
    solved = False
    
    while index < NUM_GENERATIONS:
        # Score and select population
        batch_states = []
        
        for individual in population:
            cube = Cube()
            cube.from_string(scrambled_str)
            cube.move_list(individual)
            if cube.is_solved():
                solved = True
                best_individual = individual
                sol_length = len(best_individual)
                break
            
            batch_states.append(cube.state)
            del cube
            
        fitness_score = compute_fitness(batch_states)
        scored_population = []
        
        if solved:
            break
        
        # Create Beam_Node instances using the fitness scores
        for individual, score in zip(population, fitness_score):
            scored_population.append((individual, -score))
        
        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)

        print(f"Generation {index} completed, Best Fitness: {best_fitness}")
        
        new_population = []        
        
        # Elitism: carry over top performers unchanged
        elite_individuals = elitism(scored_population, EliteRate)
                 
        selected_population = tournament_selection(population, scored_population, TournamentSize)
        
        elite_size = len(elite_individuals)

        while len(new_population) < (POPULATION_SIZE - elite_size):
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
    from scramble100 import selected_scrambles
    
    test_str = selected_scrambles[0]

    cube = Cube()
    cube.move_list(cube.convert_move(test_str))

    # # iteration 1
    succeed, generations, sol_length, best_individual = genetic_algorithm(cube.to_string(), 39000, 1000, 24, 10, 0.004)
    