import time
import torch
from cube import Cube, get_allowed_moves, load_model, device, Move
import numpy as np
from queue import PriorityQueue
# import tracemalloc
from collections import namedtuple
    
nnet = load_model()

Astar_Node = namedtuple("Astar_Node", ["cube", "moves", "g", "f"])

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

def astar_search_pq(scrambled_cube : Cube, N, scale_factor = 0.6) -> dict:
    
    time_start = time.time()
    node_explored = 1
    
    if scrambled_cube.is_solved():
        return {"success" : True, "solutions": [], "num_nodes": 1, "time_taken": 0}
    
    open_list = PriorityQueue()
    closed_list = set()
    cube_to_fitness = {}

    initial_g = 0
    initial_f = compute_fitness([scrambled_cube.state])[0] + (scale_factor * initial_g)
    start_node = Astar_Node(scrambled_cube.to_string(), [], initial_g, initial_f)
    
    open_list.put((start_node.f, start_node))
    
    cube_to_fitness[scrambled_cube.__hash__()] = initial_f

    iteration = 0
    while not open_list.empty():
        
        best_nodes = []
        batch_info = []
        batch_states = []
        
        # Collect batch
        while not open_list.empty() and len(best_nodes) < N:
            _, current_node = open_list.get()
            if current_node.cube in closed_list:
                continue

            closed_list.add(current_node.cube)
            best_nodes.append(current_node)
            
        # print(f"Iteration: {iteration}, Node Explored: {node_explored}, Best F: {best_nodes[0].f}")
                
        # Generate new states for the batch
        for node in best_nodes:
            allowed_moves = get_allowed_moves(node.moves)
            
            for move in allowed_moves:
                new_moves = node.moves + [move]
                tempcube = Cube()
                tempcube.from_string(node.cube)
                tempcube.move(move)
            
                if tempcube.is_solved():
                    return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_explored, "time_taken": time.time() - time_start}
                
                batch_states.append(tempcube.state)
                batch_info.append((tempcube.to_string(), new_moves, node.g , tempcube.__hash__()))
                
                del tempcube

        # Convert batch_states to numpy array and compute fitness
        fitness_scores = compute_fitness(batch_states)

        for ((cube_str, new_moves, g, cube_hash), fitness) in zip(batch_info, fitness_scores):
            updated_g = g + 1
            updated_f = (scale_factor * updated_g) + fitness
            new_node : Astar_Node = Astar_Node(cube_str, new_moves, updated_g, updated_f)
            
            score = cube_to_fitness.get(cube_hash)
            if score and score <= new_node.f:
                continue
            
            cube_to_fitness[cube_hash] = new_node.f
            open_list.put((new_node.f, new_node))
            
            node_explored += 1
        iteration += 1
                  
    return {"success" : False, "solutions": None, "num_nodes": node_explored, "time_taken": time.time() - time_start}

if __name__ == "__main__":
    from scramble100 import selected_scrambles
    
    success = 0
    total_sol_length = 0
    total_nodes = 0
    total_time = 0
    
    for i, scramble in enumerate(selected_scrambles):
        print(f"Test {i + 1}")
        cube = Cube()
        cube.move_list(cube.convert_move(scramble))
        
        # tracemalloc.start()
        result = astar_search_pq(cube, 3000, 0.6)
  
        # _, max = tracemalloc.get_traced_memory()
        # print(f"Peak memory usage: {max / 10**6}MB")
        # tracemalloc.stop()
        
        if result["success"]:
            success += 1
            total_sol_length += len(result["solutions"])
            total_nodes += result["num_nodes"]
            total_time += result["time_taken"]
            cube.move_list(result["solutions"])
            
            print(f"Success: {success}, Sol Length: {len(result['solutions'])}, Num Nodes: {result['num_nodes']}, Time Taken: {result['time_taken']}, Check: {cube.is_solved()}")
    
    print(f"Success Rate: {success/100}, Avg Sol Length: {total_sol_length/success}, Avg Num Nodes: {total_nodes/success}, Avg Time Taken: {total_time/success}")