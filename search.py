from cube import Cube, load_model, device, get_allowed_moves
import numpy as np
import torch
import time
from collections import namedtuple
from heapq import heappush, heappop
from typing import Union

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

class GBFS:
    def __init__(self, start_cube : Union[Cube, None] = None, batch_size = 1, max_depth = 40):
        self.start_cube = start_cube
        self.batch_size = batch_size
        self.max_depth = max_depth

    
    def search(self) -> dict:
        
        print(f"Starting GBFS Search with N = {self.batch_size} and max depth = {self.max_depth}")
        
        assert self.start_cube is not None, "Start cube is not set"
        
        node_explored = 1
        start_time = time.time()
        open = []
        visited = set()
        
        if self.start_cube.is_solved():
            return {"success" : True, "solutions": [], "num_nodes": 1, "time_taken": 0}
        
        initial_f = (compute_fitness([self.start_cube.state])[0])
        start_node = Node(self.start_cube, [], 0, initial_f, self.start_cube.is_solved(), 0)
        
        heappush(open, (start_node.f, start_node))
        visited.add(self.start_cube.__hash__())
        
        iteration = 0
        while open:
            
            if iteration == self.max_depth:
                return {"success" : False}
                 
            best_nodes = []
            batch_info = []
            batch_states = []
            
            while open and len(best_nodes) < self.batch_size:
                _, current_node = heappop(open)
                
                visited.add(current_node.cube.__hash__())
                best_nodes.append(current_node)
                        
            for current_node in best_nodes:

                # expand node
                allowed_moves = get_allowed_moves(current_node.moves)
                    
                for move in allowed_moves:
                    new_moves = current_node.moves + [move]
                    tempcube = current_node.cube.copy()
                    tempcube.move(move)
                
                    if tempcube.is_solved():
                        return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_explored, "time_taken": time.time() - start_time}
                    
                    if tempcube.__hash__() in visited:
                        continue
                    
                    batch_states.append(tempcube.state)
                    batch_info.append((tempcube, new_moves))
                    
                    del tempcube

            # Convert batch_states to numpy array and compute fitness
            fitness_scores = compute_fitness(batch_states)

            for ((temp_cube, new_moves), fitness) in zip(batch_info, fitness_scores):                
                new_node : Node = Node(temp_cube, new_moves, 0, fitness[0], False, 0)
                
                heappush(open, (new_node.f, new_node))
                visited.add(temp_cube.__hash__())
                node_explored += 1
                
            iteration += 1
            
        return {"success" : False}
    
    def __str__(self) -> str:
        return f"GBFS(batch_size={self.batch_size}, max_depth={self.max_depth})"

class WAStar:
    def __init__(self, start_cube : Union[None, Cube] = None, scale_factor=1.5, batch_size=1):
        self.scale_factor = scale_factor
        self.start_cube = start_cube
        self.batch_size = batch_size
        
    def search(self) -> dict:
        print(f"Starting WAStar Search with scale factor = {self.scale_factor} and batch size = {self.batch_size}")
        
        assert self.start_cube is not None, "Start cube is not set"
        
        time_start = time.time()
        node_explored = 1
        open_list = []
        close = set()
        cube_to_steps = {} # open union close with g value

        if self.start_cube.is_solved():
            return {"success": True, "solutions": [], "num_nodes": 1, "time_taken": 0}

        initial_g = 0
        initial_f = (compute_fitness([self.start_cube.state])[0] * self.scale_factor) + initial_g
        start_node = Node(self.start_cube.to_string(), [], initial_g, initial_f, self.start_cube.is_solved(), 0)

        heappush(open_list, (start_node.f, start_node))
        cube_to_steps[self.start_cube.__hash__()] = 0
        
        while open_list:
            best_nodes = []
            batch_info = []
            batch_states = []

            # Collect batch
            while open_list and len(best_nodes) < self.batch_size:
                _, current_node = heappop(open_list)
                cube_to_steps[current_node.cube.__hash__()] = current_node.g
                
                close.add(current_node.cube)
                best_nodes.append(current_node)

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
                    batch_info.append((tempcube.to_string(), new_moves, node.g, tempcube.__hash__()))

                    del tempcube

            # Compute fitness for batch states
            fitness_scores = compute_fitness(batch_states)

            for ((cube_str, new_moves, g, cube_hash), fitness) in zip(batch_info, fitness_scores):
                updated_g = g + 1
                updated_f = updated_g + (self.scale_factor * fitness[0])
                new_node = Node(cube_str, new_moves, updated_g, updated_f, False, 0)

                score = cube_to_steps.get(cube_hash)
                
                if not score or score > new_node.g:
                    if cube_hash in close:
                        close.remove(cube_hash)
                    cube_to_steps[cube_hash] = new_node.g
                    heappush(open_list, (updated_f, new_node))
                    node_explored += 1

        return {"success": False}

    def __str__(self) -> str:
        return f"WAStar(scale_factor={self.scale_factor}, batch_size={self.batch_size})"

class BeamSearch:
    def __init__(self, start_cube : Union[Cube, None] = None, beam_width = 1000, max_depth = 40, adaptive = False, scale_factor = 0.5):

        self.start_cube = start_cube
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.adaptive = adaptive
        self.scale_factor = scale_factor

        
    def search(self) -> dict:
        
        print(f"Starting Beam Search with beam width = {self.beam_width}, max depth = {self.max_depth}, scale factor = {self.scale_factor} and adaptive = {self.adaptive}")
            
        assert self.start_cube is not None, "Start cube is not set"    
        
        seen = set()        
        node_searched = 1
        start_time = time.time()
        seen.add(self.start_cube.__hash__())
        
        initial_g = 0
        root_fitness = compute_fitness([self.start_cube.state])[0][0] + initial_g * self.scale_factor
        root = Node(self.start_cube.to_string(), [], initial_g, root_fitness, self.start_cube.is_solved(), 0)
        
        if root.is_solved:
            return {"success" : True, "solutions": [], "length": 0, "num_nodes": 1, "time_taken": 0}
        
        generation = [root]

        for depth in range(self.max_depth + 1):
                
            if depth == self.max_depth:
                return {"success" : False}
                    
            new_generation, searched_nodes, success = self.generate_new_generation(generation, seen)
                    
            if success:
                return {"success" : True, "solutions": new_generation[0].moves, "length": len(new_generation[0].moves), "num_nodes": node_searched, "time_taken": time.time() - start_time}
            
            node_searched += searched_nodes
            
            adaptive_beam_width = self.beam_width
            
            if self.adaptive:
                adaptive_beam_width = int(self.beam_width * (0.985 ** depth))
            
            generation = new_generation[: adaptive_beam_width]
                        
        return {"success" : False}

        
    def generate_new_generation(self, generation: list[Node], seen: set) -> tuple:
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
            
                if tempcube.is_solved():
                    return [Node(tempcube.to_string(), new_moves, 0, 0, True, 0)], nodes_searched + 1, True

                if tempcube.__hash__() in seen:
                    continue

                batch_states.append(tempcube.state)
                batch_info.append((tempcube.to_string(), new_moves, node.g))
                seen.add(tempcube.__hash__())
                nodes_searched += 1
                
                del tempcube
                
        fitness_scores = compute_fitness(batch_states)
        for (cube_str, new_moves, g), fitness in zip(batch_info, fitness_scores):
            updated_g = g + 1
            updated_fitness = fitness[0] + updated_g * self.scale_factor
            new_generation.append((Node(cube_str, new_moves, updated_g, updated_fitness, False, 0)))

        new_generation.sort(key=lambda x: x.f)
        return new_generation, nodes_searched, False
    
    def __str__(self) -> str:
        return f"BeamSearch(beam_width={self.beam_width}, max_depth={self.max_depth}, adaptive={self.adaptive}, scale_factor={self.scale_factor})"

class AWA:
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

def test(search_algos : list):
    from scramble100 import selected_scrambles
    
    # setting up the initial values
    vals = {}
    for algo in search_algos:
        vals[str(algo)] = {"success": 0, "total_sol_length": 0, "total_nodes": 0, "total_time": 0}

    for i, scramble in enumerate(selected_scrambles):
        print(f"Test {i + 1}")
        for algo in search_algos:
            cube = Cube()
            cube.move_list(cube.convert_move(scramble))
            
            algo.start_cube = cube.copy()
            
            result = algo.search()
            
            if result["success"]:
                vals[str(algo)]["success"] += 1
                vals[str(algo)]["total_sol_length"] += len(result["solutions"])
                vals[str(algo)]["total_nodes"] += result["num_nodes"]
                vals[str(algo)]["total_time"] += result["time_taken"]
                
                cube.move_list(result["solutions"])
                
                print(f"{str(algo)}: Success! Solution Length: {len(result['solutions'])}, Nodes: {result['num_nodes']}, Time: {result['time_taken']}, Sanity Check: {cube.is_solved()}")
                
        print()

    print("Results:")
    # calculate the average values
    for algo in search_algos:
        if vals[str(algo)]["success"] > 0:
            vals[str(algo)]["avg_sol_length"] = vals[str(algo)]["total_sol_length"] / vals[str(algo)]["success"]
            vals[str(algo)]["avg_nodes"] = vals[str(algo)]["total_nodes"] / vals[str(algo)]["success"]
            vals[str(algo)]["avg_time"] = vals[str(algo)]["total_time"] / vals[str(algo)]["success"]
        else:
            vals[str(algo)]["avg_sol_length"] = 0
            vals[str(algo)]["avg_nodes"] = 0
            vals[str(algo)]["avg_time"] = 0
        
        print(f"{str(algo)}: Success:{vals[str(algo)]['success']}, Average Solution Length: {vals[str(algo)]['avg_sol_length']}, Average Nodes: {vals[str(algo)]['avg_nodes']}, Average Time: {vals[str(algo)]['avg_time']}")

if __name__ == "__main__":    
    search_list = [
        # WAStar(None, 1.5, 500),
        # WAStar(None, 2, 500),
        # WAStar(None, 2.5, 500),
        # WAStar(None, 3, 500),
        # WAStar(None, 3.5, 500),
        WAStar(None, 4, 500),
        WAStar(None, 5, 500),
        WAStar(None, 8, 500),
        WAStar(None, 10, 500)
    ]

    test(search_list)    

