import time
import torch
from cube import Cube, load_model, get_allowed_moves, device
import numpy as np
from queue import PriorityQueue
from collections import namedtuple
# import tracemalloc

nnet = load_model()

Beam_Node = namedtuple("Beam_Node", ["cube", "moves", "fitness", "is_solved"])
        
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

def generate_new_generation(generation: list[Beam_Node], seen_state):
    new_generation = []
    nodes_searched = 0
    batch_states = []
    batch_info = []  # To keep track of the corresponding cube and moves
    
    for node in generation:
        allowed_moves = get_allowed_moves(node.moves)
        
        for move in allowed_moves:            
            new_moves = node.moves + [move]
            tempcube = Cube()
            tempcube.from_string(node.cube)
            tempcube.move(move)
            
            if tempcube in seen_state:
                continue
        
            if tempcube.is_solved():
                return [Beam_Node(tempcube.to_string(), new_moves, 0, True)], nodes_searched, True

            batch_states.append(tempcube.state)
            batch_info.append((tempcube.to_string(), new_moves))
            
            del tempcube
            
            nodes_searched += 1

    fitness_scores = compute_fitness(batch_states)
    for (cube_str, new_moves), fitness in zip(batch_info, fitness_scores):
        new_generation.append(Beam_Node(cube_str, new_moves, fitness, False))

    # Sort new generation based on fitness
    new_generation.sort(key=lambda x: x.fitness)
    return new_generation, nodes_searched, False

def beam_search(scrambled_cube : Cube, beam_width = 1024, max_depth = 100) -> dict:
    root_fitness = compute_fitness([scrambled_cube.state])[0]
    root = Beam_Node(scrambled_cube.to_string(), [], root_fitness, scrambled_cube.is_solved())
    
    if root.is_solved:
        return {"success" : True, "solutions": [], "num_nodes": 1, "time_taken": 0}
    
    generation = [root]
    start_time = time.time()
    node_searched = 1
    
    seen_state = set()
    seen_state.add(scrambled_cube)
    
    for depth in range(max_depth + 1):
              
        if depth == max_depth:
            return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}
                
        new_generation, searched_nodes, success = generate_new_generation(generation, seen_state)
        
        if success:
            return {"success" : True, "solutions": new_generation[0].moves, "num_nodes": node_searched + searched_nodes, "time_taken": time.time() - start_time}
        
        node_searched += searched_nodes
        generation = new_generation[: beam_width]
        
        for node in generation:
            seen_state.add(node.cube)

    return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}

if __name__ == "__main__":
    from scramble100 import selected_scrambles
    
    success = 0
    total_sol_length = 0
    total_nodes = 0
    total_time = 0

    # limit = 39000
    """
    1024 =>
    Success Rate: 0.68, Avg Sol Length: 25.235294117647058, Avg Num Nodes: 298359.92647058825, Avg Time Taken: 61.83388700204737
    
    1911 =>
    Success Rate: 0.85, Avg Sol Length: 25.0, Avg Num Nodes: 541414.0352941176, Avg Time Taken: 116.34260092623093
    
    2048 =>
    Success Rate: 0.88, Avg Sol Length: 24.59090909090909, Avg Num Nodes: 566032.8863636364, Avg Time Taken: 110.20973552898927
    
    4096 =>
    Success Rate: 0.98, Avg Sol Length: 24.183673469387756, Avg Num Nodes: 1087275.0204081633, Avg Time Taken: 214.24532394506494    
    
    Memory Usage:
    3000 => Peak memory usage: 400 ~ 500MB
    
    """
        
    for i, scramble in enumerate(selected_scrambles):
        print(f"Test {i + 1}")
        cube = Cube()
        cube.move_list(cube.convert_move(scramble))
        
        # tracemalloc.start()
        result = beam_search(cube, 3000, 100)
        
        # _, max_mem = tracemalloc.get_traced_memory()
        # print(f"Peak memory usage: {max_mem / 10**6}MB")
        
        # tracemalloc.stop()
        
        if result["success"]:
            success += 1
            total_sol_length += len(result["solutions"])
            total_nodes += result["num_nodes"]
            total_time += result["time_taken"]
            cube.move_list(result["solutions"])
            print(f"Success: {success}, Sol Length: {len(result['solutions'])}, Num Nodes: {result['num_nodes']}, Time Taken: {result['time_taken']}, Check: {cube.is_solved()}")
    
    print(f"Success Rate: {success/100}, Avg Sol Length: {total_sol_length/success}, Avg Num Nodes: {total_nodes/success}, Avg Time Taken: {total_time/success}")