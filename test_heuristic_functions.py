from cube import Cube, load_model, Move, get_allowed_moves, inverse_moves, device
import numpy as np
import torch
import random
import scramble100
import matplotlib.pyplot as plt

nnet = load_model()
nnet1 = load_model("reversed")

@ torch.no_grad()
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

@ torch.no_grad()
def compute_prob(states):
    # Convert list of states to a NumPy array (if not already an array)
    states_np = np.array(states)
    # Convert NumPy array to a PyTorch tensor
    input_tensor = torch.tensor(states_np).to(device)
    # Compute output in a single batch
    outputs = nnet1(input_tensor)
    
    batch_p1 = torch.nn.functional.softmax(outputs, dim=1)
    batch_p1 = batch_p1.detach().cpu().numpy()
    torch.cuda.empty_cache()  # Clear CUDA cache here
    return batch_p1

@torch.no_grad()
def test_original_admissibility(trials=10000):
    cube = Cube()
    failed = 0
    trials = trials
    
    failed_dict = {}

    for _ in range(trials):
        # choose a random scramble from the 1000 scrambles
        random_index = random.randint(0, 999)
        # choose a random scramble length from this scramble
        moves = cube.convert_move(scramble100.scrambles[random_index])
        random_length = random.randint(1, len(moves) - 1)
        scramble = moves[:random_length]
                
        cube.reset()
        cube.move_list(scramble)
        
        heuristic_value = compute_fitness([cube.state])[0][0]
        if heuristic_value > random_length:
            if random_length not in failed_dict:
                failed_dict[random_length] = 0
                
            failed_dict[random_length] += 1
            
            failed += 1
            
    print(f"Fail rate: {failed / trials * 100}%")
    # create histogram from the failed_dict
    plt.bar(*zip(*failed_dict.items()))
    
    # save the histogram
    plt.savefig("original_admissibility.png")
    # clear the plot
    plt.clf()
    

@torch.no_grad()
def test_enhanced_admissibility(trials=10000):
    cube = Cube()
    failed = 0
    trials = trials

    failed_dict = {}
    
    for _ in range(trials):
        # choose a random scramble from the 1000 scrambles
        random_index = random.randint(0, 999)
        # choose a random scramble length from this scramble
        moves = cube.convert_move(scramble100.scrambles[random_index])
        random_length = random.randint(2, len(moves) - 1)
        scramble = moves[:random_length]
                
        cube.reset()
        cube.move_list(scramble)
        
        p = compute_prob([cube.state])[0]
        last_move = inverse_moves[scramble[-1]]
        
        p = p[last_move]
        
        cube.move(last_move)
        fitness = compute_fitness([cube.state])[0][0]
        
        heuristic_value = fitness - p
        
        # if heuristic_value is in [random_length -0.5, random_length + 0.5] then it is admissible    
        if heuristic_value > random_length - 1:
            if random_length not in failed_dict:
                failed_dict[random_length] = 0
                
            failed_dict[random_length] += 1
            
            failed += 1

    print(f"Fail rate: {failed / trials * 100}%")
    plt.bar(*zip(*failed_dict.items()))

    plt.savefig("enhanced_admissibility.png")
    plt.clf()

@torch.no_grad()
def test_consistency_original(trails=10000):
    cube = Cube()
    inconsistencies = 0
    trials = trails
    inconsistencies_dict = {}

    for _ in range(trials):
        # choose a random scramble from the 1000 scrambles
        random_index = random.randint(0, 999)
        # choose a random scramble length from this scramble
        moves = cube.convert_move(scramble100.scrambles[random_index])
        random_length = random.randint(1, len(moves) - 1)
        scramble = moves[:random_length]
                
        cube.reset()
        cube.move_list(scramble)
        original_fitness = compute_fitness([cube.state])[0][0]
        
        moves = get_allowed_moves(scramble)
        move = random.choice(moves)
        cube.move(move)
        
        successor_fitness = compute_fitness([cube.state])[0][0]
        
        # Check consistency: h(n) should be less than or equal to h(n') + 1 (cost of one move)
        if original_fitness > successor_fitness + 1:
            if random_length not in inconsistencies_dict:
                inconsistencies_dict[random_length] = 0
                
            inconsistencies_dict[random_length] += 1
            
            inconsistencies += 1

    print(f"Fail rate: {inconsistencies / trials * 100}%")
    plt.bar(*zip(*inconsistencies_dict.items()))
    plt.savefig("original_consistency.png")
    plt.clf()


@torch.no_grad()
def test_consistency_enhanced(trials=10000):
    cube = Cube()
    inconsistencies = 0
    trials = trials
    inconsistencies_dict = {}

    for _ in range(trials):
        # choose a random scramble from the 1000 scrambles
        random_index = random.randint(0, 999)
        # choose a random scramble length from this scramble
        moves = cube.convert_move(scramble100.scrambles[random_index])
        random_length = random.randint(2, len(moves) - 1)
        scramble = moves[:random_length]

        # get the original heuristic value
        cube.reset()
        cube.move_list(scramble)
        original_probs = compute_prob([cube.state])[0]
        last_move = inverse_moves[scramble[-1]]
        cube.move(last_move)
        original_fitness = compute_fitness([cube.state])[0][0]
        original_prob = original_probs[last_move]
        original_heuristic = original_fitness - original_prob
        
        
        # get the heuristic value of the successor state
        new_prob = compute_prob([cube.state])[0]
        moves = get_allowed_moves(scramble[:-1])
        move = random.choice(moves)
        cube.move(move)
        
        successor_fitness = compute_fitness([cube.state])[0][0]
        successor_prob = new_prob[move]
        successor_heuristic = successor_fitness - successor_prob
        
        # Check consistency: h(n) should be less than or equal to h(n') + 1 (cost of one move)
        if original_heuristic > successor_heuristic + 1:
            if random_length not in inconsistencies_dict:
                inconsistencies_dict[random_length] = 0
                
            inconsistencies_dict[random_length] += 1
                
            inconsistencies += 1
        

    print(f"Fail rate: {inconsistencies / trials * 100}%")
    plt.bar(*zip(*inconsistencies_dict.items()))
    plt.savefig("enhanced_consistency.png")
    plt.clf()
        
def test_goal_awareness():
    cube = Cube()
    cube.reset()
    
    # Test the goal awareness of the model
    if compute_fitness([cube.state])[0][0] <= 0:
        print("Original model is goal-aware.")
        
    moves = list(Move)
    for move in moves:
        cube.move(move)
        p = compute_prob([cube.state])[0]
        p = p[inverse_moves[move]]
        
        cube.move(inverse_moves[move])
        fitness = compute_fitness([cube.state])[0][0]
        updated_fitness = fitness - p
        
        if updated_fitness > 0:
            print("Model is not goal-aware")
            break
        
        cube.reset()
        
    print("Combined model is goal-aware.")
    
    
def test_computing_speed(trials=10000):
    cube = Cube()
    trials = trials
    states = []
    
    for _ in range(trials):
        random_index = random.randint(0, 999)
        moves = cube.convert_move(scramble100.scrambles[random_index])
        random_length = random.randint(1, len(moves) - 1)
        scramble = moves[:random_length]
                
        cube.reset()
        cube.move_list(scramble)
        states.append(cube.state)
        
    import time
    start = time.time()
    compute_fitness(states)
    end = time.time()
    print(f"Original model: {end - start} seconds")
    
    size_10 = trials // 10
    prob_state = states[:size_10]
    start = time.time()
    compute_prob(prob_state)
    compute_fitness(states)
    end = time.time()
    print(f"Enhanced model: {end - start} seconds")


if __name__ == "__main__":
    # Test Admissibility
    test_original_admissibility(10000)
    test_enhanced_admissibility(10000)
    
    # Test Consistency
    test_consistency_original(10000)
    test_consistency_enhanced(10000)
    
    # Test Goal Awareness
    test_goal_awareness()
    
    # Test Computing Speed
    test_computing_speed(30000)