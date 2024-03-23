import time
import torch
from cube import Cube, get_allowed_moves, load_model, device, Move
import numpy as np
import logging
from queue import PriorityQueue
    
nnet = load_model()

class Astart_Node:
    def __init__(self, cube : Cube, moves : list[Move]):
        self.cube = cube
        self.moves = moves
        self.g = 0
        self.h = 0
        self.f = 0
        self.parent = None
        
    def get_possible_moves(self):
        return get_allowed_moves(self.moves)
    
    def is_solved(self):        
        return cube.is_solved()
    
    def update_f(self):
        self.f = 0.6 * self.g + self.h
        
    def __eq__(self, other):
        return self.cube == other.cube
    
    def __lt__(self, other):
        return self.f < other.f
    

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

def astar_search_pq(scrambled_cube : Cube, N) -> dict:
    
    time_start = time.time()
    node_explored = 1
    
    open_list = PriorityQueue()
    closed_list = set()
    cube_to_node = {}  # Hash table for quick lookup

    initial_h = compute_fitness([scrambled_cube.state])[0]
    start_node = Astart_Node(scrambled_cube, [])
    start_node.h = initial_h
    start_node.update_f()
    
    open_list.put((start_node.f, start_node))
    cube_to_node[scrambled_cube] = start_node
    
    logging.info("Starting search with scrambled cube state.")

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
        
        logging.info(f"Best node: {best_nodes[0].f} with moves: {best_nodes[0].moves}")
            
        # Generate new states for the batch
        for node in best_nodes:
            allowed_moves = node.get_possible_moves()
            
            for move in allowed_moves:
                new_moves = node.moves + [move]
                tempcube = node.cube.copy()
                tempcube.move(move)
            
                if tempcube.is_solved():
                    return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_explored, "time_taken": time.time() - time_start}
                
                batch_states.append(tempcube.state)
                batch_info.append((tempcube, new_moves, node))
                        
        # Convert batch_states to numpy array and compute fitness
        batch_states_np = np.array(batch_states)
        fitness_scores = compute_fitness(batch_states_np)
        
        for ((cube, new_moves, parent), fitness) in zip(batch_info, fitness_scores):
            new_node = Astart_Node(cube, new_moves)
            new_node.g = parent.g + 1
            new_node.h = fitness
            new_node.parent = parent
            new_node.update_f()
            
            existing_node = cube_to_node.get(cube)
            if existing_node and existing_node.f <= new_node.f:
                continue
            
            cube_to_node[cube] = new_node
            open_list.put((new_node.f, new_node))
            node_explored += 1
                
        print("Node Searched so far: ", node_explored)
        
    return {"success" : False, "solutions": None, "num_nodes": node_explored, "time_taken": time.time() - time_start}


def astar_search(scrambled_cube : Cube, N) -> dict:
    
    node_searched = 1
    start_time = time.time()
    
    open_list : list[Astart_Node] = []
    closed_list : list[Astart_Node] = []
    
    initial_h = compute_fitness([scrambled_cube.state])[0]
    start_node = Astart_Node(scrambled_cube, [])
    start_node.h = initial_h
    start_node.update_f()
    
    open_list.append(start_node)
    
    logging.info("Starting search with scrambled cube state.")

    while open_list:        
        # Sort open_list by f-value and select the N best nodes
        best_nodes = sorted(open_list, key=lambda x: x.f)[:N]

        logging.info(f"Best node: {best_nodes[0].f}")

        batch_states = []
        batch_info = []

        for node in best_nodes: 
            open_list.remove(node)
            closed_list.append(node)
            
            allowed_moves = node.get_possible_moves()
           
            for move in allowed_moves:
                new_moves = node.moves + [move]
                tempcube = node.cube.copy()
                tempcube.move(move)
            
                if tempcube.is_solved():
                    return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_searched, "time_taken": time.time() - start_time}
                
                batch_states.append(tempcube.state)
                batch_info.append((tempcube, new_moves, node))
                        
        # Convert batch_states to numpy array and compute fitness
        batch_states_np = np.array(batch_states)
        fitness_scores = compute_fitness(batch_states_np)

        for (cube, new_moves, parent), fitness in zip(batch_info, fitness_scores):
            new_node = Astart_Node(cube, new_moves)
            new_node.g = parent.g + 1
            new_node.h = fitness
            new_node.update_f()
            new_node.parent = parent
            
            for open_node in open_list:
                if open_node.cube == new_node.cube and open_node.f <= new_node.f:
                    continue
                
            for closed_node in closed_list:
                if closed_node.cube == new_node.cube:
                    if closed_node.f <= new_node.f:
                        continue
                    else:
                        closed_list.remove(closed_node)
                
            open_list.append(new_node)
            node_searched += 1
    
        print("Node Searched so far: ", node_searched)
        
    return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}

if __name__ == "__main__":
    from scramble100 import scrambles
    
    # [<Move.B1: 15>, <Move.F3: 8>, <Move.L1: 12>, <Move.F3: 8>, <Move.R3: 5>, <Move.U1: 0>, <Move.F1: 6>, <Move.D3: 11>, <Move.L3: 14>, <Move.L3: 14>, <Move.D3: 11>, <Move.R3: 5>, <Move.L1: 12>, <Move.B1: 15>, <Move.L1: 12>, <Move.F3: 8>, <Move.B3: 17>, <Move.U3: 2>, <Move.D1: 9>]
    test_str = 0
    
    cube = Cube()
    cube.move_list(cube.convert_move(scrambles[test_str]))
    
    start_time = time.time()
    result = astar_search(cube, 3000)
    print(result)
    
    # success = 0
    # total_sol_length = 0
    # total_nodes = 0
    # total_time = 0
    
    # for i, scramble in enumerate(selected_scrambles):
    #     print(f"Test {i + 1}")
    #     cube = Cube()
    #     cube.move_list(cube.convert_move(scramble))
        
    #     result = astar_search(cube.to_string(), 3000)
    #     if result["success"]:
    #         success += 1
    #         total_sol_length += len(result["solutions"])
    #         total_nodes += result["num_nodes"]
    #         total_time += result["time_taken"]
    #         cube.move_list(result["solutions"])
    #         print(f"Success: {success}, Sol Length: {len(result['solutions'])}, Num Nodes: {result['num_nodes']}, Time Taken: {result['time_taken']}, Check: {cube.is_solved()}")
    
    # print(f"Success Rate: {success/100}, Avg Sol Length: {total_sol_length/success}, Avg Num Nodes: {total_nodes/success}, Avg Time Taken: {total_time/success}")