from cube import Cube, load_model, device, get_allowed_moves, Move
import numpy as np
import torch
import time
from collections import namedtuple
from heapq import heappush, heappop

nnet = load_model()

Node = namedtuple("Node", ["cube", "moves", "g", "f", "is_solved", "wf"])

        
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
    def __init__(self, start_cube : Cube, w_admissible = 1.5, weighted = 2.0, batch_size = 1, focal = False):
        self.start_cube = start_cube
        self.w_admissible = w_admissible
        self.weighted = weighted
        self.batch_size = batch_size
        
    def calculate_potential(self, cube):
        return compute_fitness([cube.state])[0][0]
    
    def search(self) -> dict:
        
        print(f"Starting DPS Search with w_admissible = {self.w_admissible} and weighted = {self.weighted}")
        time_start = time.time()
        node_explored = 1
            
        if self.start_cube.is_solved():
            return {"success" : True, "solutions": [], "num_nodes": 1, "time_taken": 0}
        
        initial_g = 0
        initial_f = ((compute_fitness([self.start_cube.state])[0][0]) * self.weighted)+ initial_g
        start_node = Node(self.start_cube, [], initial_g, initial_f, self.start_cube.is_solved(), initial_f)
        
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
        
        
class RWA:
    def __init__(self, start_cube : Union[Cube, None], scale_factor = 2.5, batch_size = 1, max_time = 60):
        self.start_cube = start_cube
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.max_time = max_time
        
    def search(self):
        print(f"Starting AWA Search with scale factor = {self.scale_factor} and batch size = {self.batch_size}")

        assert self.start_cube is not None, "Start cube is not set"

        start_time = time.time()

        node_explored = 1
        open_list = []
        close = set()
        cube_to_steps = {} # open union close with g value
        
        incumbent = None

        if self.start_cube.is_solved():
            return {"success": True, "solutions": [], "num_nodes": 1, "time_taken": 0}

        initial_g = 0
        initial_f = (compute_fitness([self.start_cube.state])[0]) + initial_g
        weighted_f = initial_f * self.scale_factor + initial_g
        start_node = Node(self.start_cube.to_string(), [], initial_g, initial_f, self.start_cube.is_solved(), weighted_f)
        
        heappush(open_list, (weighted_f, start_node))
        cube_to_steps[start_node.cube] = initial_g
        
        while open_list and time.time() - start_time < self.max_time:
            
            best_nodes = []
            batch_info = []
            batch_states = []
            
            # Collect batch
            while open_list and len(best_nodes) < self.batch_size:
                _, current_node = heappop(open_list)
                cube_to_steps[current_node.cube.__hash__()] = current_node.g
                
                if incumbent == None or current_node.f < incumbent.f:
                    best_nodes.append(current_node)
                    close.add(current_node.cube)
        
            for node in best_nodes:
                for move in get_allowed_moves(node.moves):
                    new_moves = node.moves + [move]
                    tempcube = Cube()
                    tempcube.from_string(node.cube)
                    tempcube.move(move)
                    batch_states.append(tempcube.state)
                    batch_info.append((tempcube.to_string(), new_moves, node.g, tempcube.__hash__(), tempcube.is_solved()))

                    del tempcube

            # Compute fitness for batch states
            fitness_scores = compute_fitness(batch_states)

            for ((cube_str, new_moves, g, cube_hash, solved), fitness) in zip(batch_info, fitness_scores):
                updated_g = g + 1
                updated_f = updated_g + (fitness[0])
                new_wf = fitness[0] * self.scale_factor + updated_g
                
                if solved:
                    if incumbent == None or updated_f < incumbent.f:
                        incumbent = Node(cube_str, new_moves, updated_g, updated_f, True, new_wf)
                        print(f"Improved incumbent with {len(incumbent.moves)} moves at time {time.time() - start_time}")
                
                new_node = Node(cube_str, new_moves, updated_g, updated_f, False, new_wf)

                score = cube_to_steps.get(cube_hash)
                
                if not score or score > new_node.g:
                    if cube_hash in close:
                        close.remove(cube_hash)
                    cube_to_steps[cube_hash] = new_node.g
                    heappush(open_list, (new_wf, new_node))
                    node_explored += 1

        if open_list:
            f_min = min(open_list, key = lambda x: x[1].f)
            f_min = f_min[1].f
                   
            if incumbent != None:
                error = incumbent.f - f_min
                return {"success": True, "solutions": incumbent.moves, "length": len(incumbent.moves), "num_nodes": node_explored, "time_taken": time.time() - start_time, "error": error}
            else:
                return {"success" : False}
        else:
            if incumbent != None:
                return {"success": True, "solutions": incumbent.moves, "length": len(incumbent.moves), "num_nodes": node_explored, "time_taken": time.time() - start_time, "error": 0}
            else:
                return {"success" : False}
            
    def __str__(self) -> str:
        return f"AWA(scale_factor={self.scale_factor}, batch_size={self.batch_size}, max_time={self.max_time})"

    
if __name__ == "__main__":
    from scramble100 import selected_scrambles
    
    test_str = selected_scrambles[0]
    
    temp = Cube()
    temp.move_list(temp.convert_move(test_str))
    
    awa = AWA(temp, 5, 1)
    
    result, error = awa.search()
    print(result, error)