from cube import Cube, load_model, device, get_allowed_moves, Move
import numpy as np
import torch
import time
from collections import namedtuple
from heapq import heappush, heappop

nnet = load_model()

Node = namedtuple("Node", ["cube", "moves", "g", "f", "is_solved"])

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
        
class DPS:
    def __init__(self, start_cube : Cube, w_admissible = 1.5, weighted = 2.0, N = 1, focal = False):
        self.open = np.array([])
        self.focal = np.array([])
        self.visited = set()
        self.start_cube = start_cube
        self.w_admissible = w_admissible
        self.weighted = weighted
        self.N = N
    
    def search(self) -> dict:
        
        print("Starting DPS Search with N = ", self.N)
        time_start = time.time()
        node_explored = 1
            
        if self.start_cube.is_solved():
            return {"success" : True, "solutions": [], "num_nodes": 1, "time_taken": 0}
        
        initial_g = 0
        initial_f = ((compute_fitness([self.start_cube.state])[0][0]) * self.weighted)+ initial_g
        start_node = Node(self.start_cube, [], initial_g, initial_f, self.start_cube.is_solved())
        
        self.open = np.append(self.open, start_node)
        self.focal = np.append(self.focal, start_node)
        
        f_min = start_node.f
        
        # while focal is not empty
        while self.focal.size > 0:
            # find f_min
            best_node_index = self.chooseNode(self.focal, f_min)
        
            node_to_expand = self.focal[best_node_index]

            # remove node from focal and open
            self.focal = np.delete(self.focal, best_node_index)
            self.open = self.open[self.open != node_to_expand]
            self.visited.add(node_to_expand.cube.__hash__())

            print(f"Node Explored: {node_explored}, f_min: {f_min}")
            
            if node_to_expand.is_solved:
                return {"success": True, "solutions": node_to_expand.moves, "length": len(node_to_expand.moves), "num_nodes": node_explored, "time_taken": time.time() - time_start}
            
            if self.open.size > 0:
                new_min = min(self.open, key = lambda x: x.f).f
                if new_min > f_min:
                    self.fixFocal(f_min, new_min)
                    f_min = new_min
                
            # expand node
            for move in get_allowed_moves(node_to_expand.moves):
                new_moves = node_to_expand.moves + [move]
                tempcube = node_to_expand.cube.copy()
                tempcube.move(move)
                
                if tempcube.is_solved():
                    return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_explored, "time_taken": time.time() - time_start}
                
                if tempcube.__hash__() in self.visited:
                    continue
                
                fitness = compute_fitness([tempcube.state])[0][0]
                updated_g = node_to_expand.g + 1
                updated_f = updated_g + (fitness * self.weighted)
                
                new_node = Node(tempcube, new_moves, updated_g, updated_f, False)
                
                self.open = np.append(self.open, new_node)
                
                if updated_f <= self.w_admissible * f_min:
                    self.focal = np.append(self.focal, new_node)
                
                self.visited.add(tempcube.__hash__())
                node_explored += 1
    
             
    def fixFocal(self, old_f_min, new_f_min):
        # for n in self.open, if f(n) > old_f_min and f(n) <= new_f_min, add n to focal
        for n in self.open:
            if n.f > old_f_min and n.f <= new_f_min:
                self.focal = np.append(self.focal, n)
    
    
    def chooseNode(self, focal, f_min):
        # Extract f, g, and h values into NumPy arrays.
        f_values = np.array([node.f for node in focal])
        g_values = np.array([node.g for node in focal])
        h_values = f_values - g_values  # Compute h if not directly available.
        
        # Compute scores for each node. Handle division by zero for h=0.
        scores = np.where(h_values != 0, (self.w_admissible * f_min - g_values) / h_values, -np.inf)
        
        best_node_index = np.argmax(scores)
        
        return best_node_index
    
class AWA:
    def __init__(self, start_cube : Cube, scale_factor = 2.5, batch_size = 1, max_time = 60):
        self.start_cube = start_cube
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.max_time = max_time
        self.time_start = time.time()
        
    def search(self):
        print(f"Starting AWA Search with scale factor = {self.scale_factor} and batch size = {self.batch_size}")

        open_list = []
        closed_list = {}

        node_explored = 1

        if self.start_cube.is_solved():
            return {"success": True, "solutions": [], "num_nodes": 1, "time_taken": 0}

        initial_g = 0
        initial_f = (compute_fitness([self.start_cube.state])[0] * self.scale_factor) + initial_g
        start_node = Node(self.start_cube.to_string(), [], initial_g, initial_f, self.start_cube.is_solved())
        
        heappush(self.open, (start_node.f, start_node))
        
        # while (len(self.open))
    

# def idastar_search(scrambled_str: str) -> dict:

#     scrambled_cube = Cube()
#     scrambled_cube.from_string(scrambled_str)
#     initial_h = compute_fitness(np.array([scrambled_cube.convert_res_input()]))[0]
#     start_node = Astart_Node(scrambled_str, [])
#     start_node.h = initial_h
#     start_node.update_f()
    
#     threshold = start_node.f

#     def search(node : Astart_Node, threshold, depth = 0, max_depth = 40):
#         logging.debug(f"Depth {depth}: Threshold: {threshold}, Current Node F: {node.f}")
        
#         if depth > 16:
#             logging.info(f"Depth: {depth}, Threshold: {threshold}, Current Node F: {node.f}")
        
#         if depth > max_depth:
#             logging.info(f"Reached maximum depth of {max_depth}.")
#             return float('inf')  # Indicating that this path didn't lead to a solution within the depth limit.
#         if node.f > threshold:
#             logging.debug(f"Depth {depth}: Node f={node.f} exceeds threshold. Returning to previous level.")
#             return node.f
#         if node.is_solved():
#             logging.debug(f"Solution found with {len(node.moves)} moves: {node.moves}")
#             return node
#         min_threshold = float('inf')
        
#         # Batch preparation for parallel fitness computation
#         cube_states = []
#         cube_info = []
#         for move in node.get_possible_moves():
#             tempcube = Cube()
#             tempcube.from_string(node.cube)
#             tempcube.move(move)
#             cube_states.append(tempcube.convert_res_input())  # Prepare the cube state for fitness computation
#             cube_info.append((tempcube.to_string(), node.moves + [move]))
            
#         # Parallel fitness computation for the batch
#         fitness_scores = compute_fitness(np.array(cube_states))
        
#         for (cube_str, moves), fitness in zip(cube_info, fitness_scores):
#             new_node = Astart_Node(cube_str, moves)
#             new_node.g = node.g + 1
#             new_node.h = fitness
#             new_node.update_f()
            
#             logging.debug(f"Depth {depth}: Created new node with f={new_node.f} from moves: {moves}.")
#             temp = search(new_node, threshold, depth=depth+1, max_depth=max_depth)  # Recursive search call
            
#             if isinstance(temp, Astart_Node):  # Solution found
#                 return temp
#             if temp < min_threshold:
#                 min_threshold = temp
                
#         return min_threshold

#     logging.info(f"Starting IDA* search with initial threshold: {threshold}")
#     while True:
#             start_time = time.time()
#             result = search(start_node, threshold)
#             if isinstance(result, Astart_Node):  # Solution found
#                 return {"success": True, "solution": result.moves, "num_moves": len(result.moves)}
#             if result == float('inf'):
#                 return {"success": False, "solution": None, "num_moves": 0}
#             logging.info(f"No solution under current threshold {threshold}. Increasing threshold to {result}. Time taken: {time.time() - start_time}")
#             threshold = result


if __name__ == "__main__":
    pass