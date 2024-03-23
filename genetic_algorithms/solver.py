from rubik54 import Cube
import random
from cube_utils import Move, move_dict
from ga_utils import boltzmann_selection, mutate, compute_fitness, \
                            crossover, generate_individual, load_model, \
                            simplify_individual, elitism, kill_by_rank, \
                            tournament_selection, uniform_crossover, \
                            two_point_crossover
                            
# Main GA Loop
def genetic_algorithm_phase1(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    population = [generate_individual(SEQUENCE_LENGTH, 1) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_individual = []
    index = 0
    sol_length = 0
    
    initial_temperature = TEMPERATURE
    temperature = initial_temperature
    cooling_rate = COOLING_RATE
    
    solved = False
    
    while index < NUM_GENERATIONS:
         # Score and select population
        scored_population = [(individual, compute_fitness(scrambled_str, individual, 1)) for individual in population]
        # scored_population = compute_fitness_population(scrambled_str, population)
        
        best_fitness = max(score for _, score in scored_population)
        best_individual = [individual for individual, score in scored_population if score == best_fitness][0]
        best_fitnesses.append(best_fitness)
                        
        if best_fitness == 20:
                solved = True
                best_individual = simplify_individual(best_individual)
                best_individual = [move for move in best_individual if move != Move.N]
                sol_length = len(best_individual)
                break
                
        selected_population = boltzmann_selection(population, scored_population, temperature)
        
        # Initialize new population with crossover and mutation
        new_population = []
        while len(new_population) < len(population):
            """ with prevention algorithm """
            
            # child1, child2 = random.sample(selected_population, 2)
            # # Mutate children and add them to the new population with simplification
            # new_population.append(simplify_individual(mutate(child1, 1)))
            # if len(new_population) < len(population):
            #     new_population.append(simplify_individual(mutate(child2, 1)))
            
            """ without prevention algorithm """
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2)
            
            # Mutate children and add them to the new population
            new_population.append(mutate(child1, 1))
            if len(new_population) < len(population):
                new_population.append(mutate(child2, 1))
            

        population = new_population  # Update population with the newly generated one
        
        # Update temperature
        temperature *= cooling_rate
    
        index += 1
    
    return solved, index, sol_length, best_individual

# Main GA Loop
def genetic_algorithm_phase2(scrambled_str, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    population = [generate_individual(SEQUENCE_LENGTH, 2) for _ in range(POPULATION_SIZE)]
    
    best_fitnesses = []
    best_individual = []
    index = 0
    sol_length = 0
    
    initial_temperature = TEMPERATURE
    temperature = initial_temperature
    cooling_rate = COOLING_RATE
    
        
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
        elite_individuals = elitism(scored_population)
                 
        # # selected_population = kill_by_rank(scored_population)
        selected_population = tournament_selection(population, scored_population)
        # selected_population = boltzmann_selection(population, scored_population, temperature)

        elite_size = len(elite_individuals)

        while len(new_population) < (POPULATION_SIZE - elite_size):
            # Randomly select two parents for crossover
            child1, child2 = random.sample(selected_population, 2)
            # parent1, parent2 = random.sample(selected_population, 2)
            # child1, child2 = uniform_crossover(parent1, parent2)
            # child1, child2 = two_point_crossover(parent1, parent2)
            
            # Mutate children and add them to the new population
            new_population.append(simplify_individual(mutate(child1, 2)))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(simplify_individual(mutate(child2, 2)))
        
        for i in elite_individuals:
            new_population.append(i)
            
        # Integrate elite individuals back into the population
        population = new_population
        
        # Update temperature
        temperature *= cooling_rate
        
        index += 1
    
    return solved, index, sol_length, best_individual

def test_single(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    
    solution = ""
    phase1_sol = ""
    
    cube = Cube()
    scramble_str = cube.randomize_n(25)
    
    print("Scrambled cube:\n", cube.to_2dstring())
    print("Scrambled moves:", scramble_str)
    print("Solving...\n")
    
    print("Running phase 1")
    solved, _, _, best_individual = genetic_algorithm_phase1(cube.to_string(), 2402, 249, 25, 10.32, 0.989)
    
    if solved:
        for move in best_individual:
            if move != Move.N:
                phase1_sol += move_dict[move] + " "
            
        print("Phase 1 Solution: ", phase1_sol)
        
        print("Running phase 2")
        cube.move_list(best_individual)
        print("Phase 1 Check: ", cube.check_phase1_solved())
        
        solved, _, _,  best_individual = genetic_algorithm_phase2(cube.to_string(), POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE)
        
        if solved:
            for move in best_individual:
                solution += move_dict[move] + " "
            print("Solution: ", solution)
    else:
        return False

test_single(8000, 1000, 23, 90, 0.989)