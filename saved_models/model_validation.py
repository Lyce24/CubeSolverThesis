from cube import Cube, get_allowed_moves, inverse_moves, Move
from validation_states import scramble_list, solution_list, validation_states
from scramble100 import scrambles
from search import compute_fitness, compute_prob
import random
import matplotlib.pyplot as plt
import torch
from math import log


def mean_squared_error(x, y):
    return sum((xi - yi) ** 2 for xi, yi in zip(x, y)) / len(x)

def cross_entropy_error(x, y):
    y = [yi + 1e-10 for yi in y]
    return -sum(xi * log(yi) for xi, yi in zip(x, y))


def validate_deepcubea_model_per_step(scrambles, n=1000):
    x_axis = list(range(1, 16))
    y_axis = []
    
    cube = Cube()

    for i in range(1, 16):
        expected_y = []
        batch = []
        for _ in range(n):
            random_scramble = random.choice(scrambles)
            random_scramble = cube.convert_move(random_scramble)            
            # select random range of moves

            random_moves_counts = i
            expected_y.append(random_moves_counts)
            cube.move_list(random_scramble[:random_moves_counts])
            batch.append(cube.state)
            cube.reset()        
            
        predicted_y = compute_fitness(batch).flatten().tolist()
    
        error = mean_squared_error(expected_y, predicted_y)
        y_axis.append(error)
        
    # create a plot
    plt.plot(x_axis, y_axis)
    plt.xlabel("Number of moves")
    plt.ylabel("Mean squared error")
    plt.title("Validation of the DeepCubeA model")
    plt.savefig("deepcubea_validation.png")
    plt.clf()
    
    return y_axis

def validate_unscrambling_model_per_step(scrambles, n=1000):
    x_axis = list(range(1, 16))
    y_axis = []
    
    cube = Cube()

    for i in range(1, 16):
        total_entropy = 0
        
        expected_prob = []
        batch = []
        for _ in range(n):
            random_scramble = random.choice(scrambles)
            random_scramble = cube.convert_move(random_scramble)
            random_moves_counts = i
            cube.move_list(random_scramble[:random_moves_counts])
            
            last_move = random_scramble[random_moves_counts - 1]
            inverse_move = inverse_moves[last_move]
            index_of_move = list(Move).index(inverse_move)
            
            # create prob distribution
            prob = [0] * 12
            prob[index_of_move] = 1
            
            expected_prob.append(prob)
            
            batch.append(cube.state)
            cube.reset()        
            
        predicted_y = compute_prob(batch).tolist()
        
        for j in range(n):
            total_entropy += cross_entropy_error(expected_prob[j], predicted_y[j])
            
        y_axis.append(total_entropy / n)

    # create a plot
    plt.plot(x_axis, y_axis)
    plt.xlabel("Scramble Depth")
    plt.ylabel("Cross Entropy Error")
    plt.title("Validation of the DSL Model")
    plt.savefig("unscrambling_validation.png")
    plt.clf()
    
    return y_axis
        
        
def validate_deepcubea_model(scrambles, n=1000):
    cube = Cube()
    expected_y = []
    batch = []
    for _ in range(n):
        random_scramble = random.choice(scrambles)
        random_scramble = cube.convert_move(random_scramble)
        
        # select random range of moves
        min_index = random.randint(0, len(random_scramble) - 1)
        max_index = random.randint(min_index, len(random_scramble) - 1)
        random_moves_counts = max_index - min_index + 1
        expected_y.append(random_moves_counts)
        
        cube.move_list(random_scramble[min_index:max_index])
        batch.append(cube.state)
        cube.reset()
        
    predicted_y = compute_fitness(batch).flatten().tolist()

    return mean_squared_error(expected_y, predicted_y)
 
def validate_unscrambling_model(scrambles, n=1000):

    cube = Cube()
    total_entropy = 0
        
    expected_prob = []
    batch = []
    for _ in range(n):
        random_scramble = random.choice(scrambles)
        random_scramble = cube.convert_move(random_scramble)
        
        min_index = random.randint(0, len(random_scramble) - 1)
        max_index = random.randint(min_index, len(random_scramble) - 1)
        cube.move_list(random_scramble[min_index:max_index])
        
        last_move = random_scramble[-1]
        inverse_move = inverse_moves[last_move]
        index_of_move = list(Move).index(inverse_move)
        
        # create prob distribution
        prob = [0] * 12
        prob[index_of_move] = 1        
        expected_prob.append(prob)
        
        batch.append(cube.state)
        cube.reset()        
        
    predicted_y = compute_prob(batch).tolist()
    
    for j in range(n):
        total_entropy += cross_entropy_error(expected_prob[j], predicted_y[j])
        
    return total_entropy / n


if __name__ == "__main__":
    print(validate_deepcubea_model_per_step(scrambles, 100000))
    print(validate_unscrambling_model_per_step(scrambles, 100000))