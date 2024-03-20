from rubik54 import Cube
import random
from utils.cube_utils import Move, move_dict
# from bayes_opt import BayesianOptimization
from utils.ga_utils import boltzmann_selection, mutate, compute_fitness,generate_individual, simplify_individual, \
                            crossover, two_point_crossover, uniform_crossover, tournament_selection, elitism, kill_by_rank, get_allowed_mutations_pre
from test import load_model

cube = Cube()
cube.from_string("UULFULFBFLDBURLUUBURUDFRRLBDBLDDBFFRBRRDLFLBFDFRUBRDLD")
print(cube.to_2dstring())

# main bfs loop

nnet = load_model()

testmove = []

def next_node(testmove):
    
    allowed_move = get_allowed_mutations_pre(testmove)
    
    if Move.N in allowed_move:
        allowed_move.remove(Move.N)
    
    best_fitness = float('-inf')
    best_move = None
    
    for i in allowed_move:
        temp = testmove.copy()
        temp.append(i)
        
        testcube = Cube()
        testcube.from_string("UULFULFBFLDBURLUUBURUDFRRLBDBLDDBFFRBRRDLFLBFDFRUBRDLD")
        
        fitness = compute_fitness(testcube.to_string(), temp, 2, nnet)
        if fitness > best_fitness:
            best_fitness = fitness
            best_move = i
        
    print(f"Best Move: {best_move}, Fitness: {best_fitness}")
    return best_move

for i in range(23):
    testmove.append(next_node(testmove))
    print(f"Test Move: {testmove}")
    print(f"Test Move Length: {len(testmove)}")
    print(f"Test Move Fitness: {compute_fitness(cube.to_string(), testmove, 2, nnet)}")
    print("")