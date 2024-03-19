from rubik54 import Cube
import random
from bayes_opt import BayesianOptimization
from utils.cube_utils import Move
from utils.ga_utils import boltzmann_selection, mutate, compute_fitness, \
                            crossover, generate_individual, load_model, \
                            simplify_individual, elitism, kill_by_rank, \
                            tournament_selection, uniform_crossover, \
                            two_point_crossover

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
        elite_individuals = elitism(scored_population, 0.007)
                 
        # selected_population = kill_by_rank(scored_population, 0.5)
        selected_population = tournament_selection(population, scored_population, 0.001)
        # selected_population = boltzmann_selection(population, scored_population, temperature)

        elite_size = len(elite_individuals)

        while len(new_population) < (POPULATION_SIZE - elite_size):
            
            """ with prevention algorithm """
            child1, child2 = random.sample(selected_population, 2)
            # Mutate children and add them to the new population with simplification
            new_population.append(simplify_individual(mutate(child1, 1)))
            if len(new_population) < len(population):
                new_population.append(simplify_individual(mutate(child2, 1)))
            
            """ without prevention algorithm """
            # parent1, parent2 = random.sample(selected_population, 2)
            # child1, child2 = uniform_crossover(parent1, parent2)
            
            # # Mutate children and add them to the new population
            # new_population.append(mutate(child1, 1))
            # if len(new_population) < len(population):
            #     new_population.append(mutate(child2, 1))
        
        for i in elite_individuals:
            new_population.append(i)
            
        # Integrate elite individuals back into the population
        population = new_population
        
        # Update temperature
        temperature *= cooling_rate
        
        index += 1
    
    return solved, index, sol_length, best_individual

def test_single(POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE):
    cube = Cube()
    cube.phase2_randomize_n(30)
    test_cube = cube.to_string()
    print("Running test")
    succeed, generations, sol_length, _ = genetic_algorithm_phase2(test_cube, POPULATION_SIZE, NUM_GENERATIONS, SEQUENCE_LENGTH, TEMPERATURE, COOLING_RATE)
    print(f"Success: {succeed}, Generations: {generations}, Solution Length: {sol_length}")
    
    return succeed

test_single(8000, 200, 23, 100, 0.989)