from cube import Cube, load_model, device, get_allowed_moves, Move
import numpy as np
import torch
import time
from collections import namedtuple
from heapq import heappush, heappop
from typing import Union

nnet = load_model()
nnet1 = load_model("reversed")

Node = namedtuple("Node", ["g", "moves", "cube", "f", "solved", "wf"])

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

class WAStar:
    def __init__(self, start_cube : Union[None, Cube] = None, scale_factor=1.5, batch_size=1):
        self.scale_factor = scale_factor
        self.start_cube = start_cube
        self.batch_size = batch_size
        
    def search(self) -> dict:
        print(f"Starting BWA* Search with scale factor = {self.scale_factor} and batch size = {self.batch_size}")
        
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
        start_node = Node(initial_g, [], self.start_cube.state, initial_f, self.start_cube.is_solved(), 0)

        heappush(open_list, (start_node.f, start_node))
        
        while open_list:
            best_nodes = []
            batch_info = []
            batch_states = []

            # Collect batch
            while open_list and len(best_nodes) < self.batch_size:
                _, current_node = heappop(open_list)
                cube_to_steps[hash(tuple(current_node.cube))] = current_node.g
                
                close.add(hash(tuple(current_node.cube)))
                best_nodes.append(current_node)

            # Generate new states for the batch
            for node in best_nodes:
                allowed_moves = get_allowed_moves(node.moves)

                for move in allowed_moves:
                    new_moves = node.moves + [move]
                    tempcube = Cube()
                    tempcube.from_state(node.cube)
                    tempcube.move(move)

                    if tempcube.is_solved():
                        if len(new_moves) > 26:
                            return {"success": False}
                        return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_explored, "time_taken": time.time() - time_start}

                    batch_states.append(tempcube.state)
                    batch_info.append((tempcube.state, new_moves, node.g, tempcube.__hash__()))

                    del tempcube

            # Compute fitness for batch states
            fitness_scores = compute_fitness(batch_states)

            for ((cube_str, new_moves, g, cube_hash), fitness) in zip(batch_info, fitness_scores):
                updated_g = g + 1
                updated_f = updated_g + (self.scale_factor * fitness[0])
                new_node = Node(updated_g, new_moves, cube_str, updated_f, False, 0)

                score = cube_to_steps.get(cube_hash)
                
                if not score or score > new_node.g:
                    if cube_hash in close:
                        close.remove(cube_hash)
                    cube_to_steps[cube_hash] = new_node.g
                    heappush(open_list, (updated_f, new_node))
                    node_explored += 1

        return {"success": False}
    
    def __str__(self) -> str:
        return f"BWA*(scale_factor={self.scale_factor}, batch_size={self.batch_size})"

class MWAStar:
    def __init__(self, start_cube : Union[None, Cube] = None, scale_factor=1.5, batch_size=1):
        self.scale_factor = scale_factor
        self.start_cube = start_cube
        self.batch_size = batch_size
        
    def search(self) -> dict:
        print(f"Starting MWA* Search with scale factor = {self.scale_factor} and batch size = {self.batch_size}")
        
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
        start_node = Node(initial_g, [], self.start_cube.state, initial_f, self.start_cube.is_solved(), 0)

        heappush(open_list, (initial_f, start_node))
                
        while open_list:
            batch_info = []
            batch_states = []

            prob_info = []
            prob_states = []

            # Collect batch
            while open_list and len(prob_states) < self.batch_size:
                _, current_node = heappop(open_list)
                cube_to_steps[hash(tuple(current_node.cube))] = current_node.g
                
                close.add(hash(tuple(current_node.cube)))
                prob_states.append(current_node.cube)
                prob_info.append(current_node)

            # Compute probability for batch states
            move_probs = compute_prob(prob_states)
            
            for (node, prob) in zip(prob_info, move_probs):
                allowed_moves = get_allowed_moves(node.moves)
                for move, p in zip(list(Move), prob):
                    if move in allowed_moves:
                        new_moves = node.moves + [move]
                        tempcube = Cube()
                        tempcube.from_state(node.cube)
                        tempcube.move(move)

                        if tempcube.is_solved():
                            if len(new_moves) > 26:
                                return {"success": False}
                            return {"success": True, "solutions": new_moves, "length": len(new_moves), "num_nodes": node_explored, "time_taken": time.time() - time_start}

                        batch_states.append(tempcube.state)
                        batch_info.append((tempcube.state, new_moves, node.g, tempcube.__hash__(), p))

                        del tempcube
                        
            # Compute fitness for batch states
            fitness_scores = compute_fitness(batch_states)

            for ((cube_str, new_moves, g, cube_hash, p), fitness) in zip(batch_info, fitness_scores):
                updated_g = g + 1
                updated_f = updated_g + (self.scale_factor * (fitness[0] - p))
                new_node = Node(updated_g, new_moves, cube_str, updated_f, False, 0)

                score = cube_to_steps.get(cube_hash)
                
                if not score or score > new_node.g:
                    if cube_hash in close:
                        close.remove(cube_hash)
                    cube_to_steps[cube_hash] = new_node.g
                    heappush(open_list, (updated_f, new_node))
                    node_explored += 1

        return {"success": False}
    
    def __str__(self) -> str:
        return f"MWA*(scale_factor={self.scale_factor}, batch_size={self.batch_size})"

class AWAStar:
    def __init__(self, start_cube : Union[Cube, None] = None, scale_factor = 2.5, batch_size = 1, max_time = 60):
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
        start_node = Node(initial_g, [], self.start_cube.state, initial_f, self.start_cube.is_solved(), weighted_f)
        
        heappush(open_list, (weighted_f, start_node))
        
        while open_list and time.time() - start_time < self.max_time:
            
            best_nodes = []
            batch_info = []
            batch_states = []
            
            # Collect batch
            while open_list and len(best_nodes) < self.batch_size:
                _, current_node = heappop(open_list)
                cube_to_steps[hash(tuple(current_node.cube))] = current_node.g
                
                if incumbent == None or current_node.f < incumbent.f:
                    best_nodes.append(current_node)
                    close.add(hash(tuple(current_node.cube)))
        
            for node in best_nodes:
                for move in get_allowed_moves(node.moves):
                    new_moves = node.moves + [move]
                    tempcube = Cube()
                    tempcube.from_state(node.cube)
                    tempcube.move(move)
                    batch_states.append(tempcube.state)
                    batch_info.append((tempcube.state, new_moves, node.g, tempcube.__hash__(), tempcube.is_solved()))

                    del tempcube

            # Compute fitness for batch states
            fitness_scores = compute_fitness(batch_states)

            for ((cube_str, new_moves, g, cube_hash, solved), fitness) in zip(batch_info, fitness_scores):
                updated_g = g + 1
                updated_f = updated_g + (fitness[0])
                new_wf = fitness[0] * self.scale_factor + updated_g
                
                if solved:
                    if incumbent == None or updated_f < incumbent.f:
                        incumbent = Node(updated_g, new_moves, cube_str, updated_f, True, new_wf)
                
                new_node = Node(updated_g, new_moves, cube_str, updated_f, False, new_wf)

                score = cube_to_steps.get(cube_hash)
                
                if not score or score > new_node.g:
                    if cube_hash in close:
                        close.remove(cube_hash)
                    cube_to_steps[cube_hash] = new_node.g
                    heappush(open_list, (new_wf, new_node))
                    node_explored += 1
                    
        if incumbent != None:
            if open_list:
                f_min = min(open_list, key = lambda x: x[1].f)
                f_min = f_min[1].f
                
                error = incumbent.f - f_min
                return {"success": True, "solutions": incumbent.moves, "length": len(incumbent.moves), "num_nodes": node_explored, "time_taken": time.time() - start_time, "error": error}
            else:
                return {"success": True, "solutions": incumbent.moves, "length": len(incumbent.moves), "num_nodes": node_explored, "time_taken": time.time() - start_time, "error": 0}
            
        return {"success" : False}
            
    def __str__(self) -> str:
        return f"AWA(scale_factor={self.scale_factor}, batch_size={self.batch_size}, max_time={self.max_time})"

class MAWAStar:
    def __init__(self, start_cube : Union[Cube, None] = None, scale_factor = 2.5, batch_size = 1, max_time = 60):
        self.start_cube = start_cube
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.max_time = max_time
        
    def search(self):
        print(f"Starting MAWA* Search with scale factor = {self.scale_factor} and batch size = {self.batch_size}")

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
        start_node = Node(initial_g, [], self.start_cube.state, initial_f, self.start_cube.is_solved(), weighted_f)
        
        heappush(open_list, (weighted_f, start_node))
        
        while open_list and time.time() - start_time < self.max_time:
            batch_info = []
            batch_states = []

            prob_info = []
            prob_states = []

            # Collect batch
            while open_list and len(prob_states) < self.batch_size:
                _, current_node = heappop(open_list)
                cube_to_steps[hash(tuple(current_node.cube))] = current_node.g
                
                if incumbent == None or current_node.f < incumbent.f:
                    close.add(hash(tuple(current_node.cube)))
                    prob_states.append(current_node.cube)
                    prob_info.append(current_node)
                    
            if not prob_states:
                break
        
            # Compute probability for batch states
            move_probs = compute_prob(prob_states)
            
            for (node, prob) in zip(prob_info, move_probs):
                allowed_moves = get_allowed_moves(node.moves)
                for move, p in zip(list(Move), prob):
                    if move in allowed_moves:
                        new_moves = node.moves + [move]
                        tempcube = Cube()
                        tempcube.from_state(node.cube)
                        tempcube.move(move)
                        batch_states.append(tempcube.state)
                        batch_info.append((tempcube.state, new_moves, node.g, tempcube.__hash__(), p, tempcube.is_solved()))

                        del tempcube
                        
            # Compute fitness for batch states
            fitness_scores = compute_fitness(batch_states)
            
            for ((cube_str, new_moves, g, cube_hash, p, solved), fitness) in zip(batch_info, fitness_scores):
                updated_g = g + 1
                updated_f = updated_g + (fitness[0])
                new_wf = updated_g + (self.scale_factor * (fitness[0] - p))
                
                if solved:
                    if incumbent == None or updated_f < incumbent.f:
                        incumbent = Node(updated_g, new_moves, cube_str, updated_f, True, new_wf)
                
                new_node = Node(updated_g, new_moves, cube_str, updated_f, False, new_wf)
                score = cube_to_steps.get(cube_hash)
                
                if not score or score > new_node.g:
                    if cube_hash in close:
                        close.remove(cube_hash)
                    cube_to_steps[cube_hash] = new_node.g
                    heappush(open_list, (new_wf, new_node))
                    node_explored += 1
                    
        if incumbent != None:
            if open_list:
                f_min = min(open_list, key = lambda x: x[1].f)
                f_min = f_min[1].f
                
                error = incumbent.f - f_min
                return {"success": True, "solutions": incumbent.moves, "length": len(incumbent.moves), "num_nodes": node_explored, "time_taken": time.time() - start_time, "error": error}
            else:
                return {"success": True, "solutions": incumbent.moves, "length": len(incumbent.moves), "num_nodes": node_explored, "time_taken": time.time() - start_time, "error": 0}
            
        return {"success" : False}
            
    def __str__(self) -> str:
        return f"MAWA(scale_factor={self.scale_factor}, batch_size={self.batch_size}, max_time={self.max_time})"

class MBS:
    def __init__(self, start_cube : Union[Cube, None] = None, beam_width = 1000, max_depth = 40, adaptive = False):
        self.start_cube = start_cube
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.adaptive = adaptive
        
    def search(self) -> dict:
        
        print(f"Starting MBS with beam width = {self.beam_width}, max depth = {self.max_depth} and adaptive = {self.adaptive}")
            
        assert self.start_cube is not None, "Start cube is not set"    
        
        seen = set()        
        node_searched = 1
        start_time = time.time()
        seen.add(self.start_cube.__hash__())
        
        root_fitness = compute_fitness([self.start_cube.state])[0][0]
        root = Node(0, [], self.start_cube.state, root_fitness, self.start_cube.is_solved(), 0)
        
        if root.solved:
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
        
        prob_state = [node.cube for node in generation]
        
        batch_states = []
        batch_info = []  # To keep track of the corresponding cube and moves
        
        probs = compute_prob(prob_state)
        
        for node, prob in zip(generation, probs):
            allowed_moves = get_allowed_moves(node.moves)
            for move, p in zip(list(Move), prob):
                if move in allowed_moves:
                    new_moves = node.moves + [move]
                    tempcube = Cube()
                    tempcube.from_state(node.cube)
                    tempcube.move(move)
                    
                    if tempcube.is_solved():
                        return [Node(0, new_moves, tempcube.state, 0, True, 0)], nodes_searched + 1, True

                    if tempcube.__hash__() in seen:
                        continue

                    batch_states.append(tempcube.state)
                    batch_info.append((tempcube.state, new_moves, p))
                    seen.add(tempcube.__hash__())
                    nodes_searched += 1

                    del tempcube
                
        fitness_scores = compute_fitness(batch_states)
        for (cube_str, new_moves, p), fitness in zip(batch_info, fitness_scores):
            updated_f = fitness[0] - p
            new_generation.append(Node(0, new_moves, cube_str, updated_f, False, 0))
            
        new_generation.sort(key=lambda x: x.f)
        return new_generation, nodes_searched, False
    
    def __str__(self) -> str:
        return f"MBS(beam_width={self.beam_width}, max_depth={self.max_depth}, adaptive={self.adaptive})"


class BeamSearch:
    def __init__(self, start_cube : Union[Cube, None] = None, beam_width = 1000, max_depth = 40, adaptive = False):

        self.start_cube = start_cube
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.adaptive = adaptive
        
    def search(self) -> dict:
        
        print(f"Starting Beam Search with beam width = {self.beam_width}, max depth = {self.max_depth} and adaptive = {self.adaptive}")
            
        assert self.start_cube is not None, "Start cube is not set"    
        
        seen = set()        
        node_searched = 1
        start_time = time.time()
        seen.add(self.start_cube.__hash__())
        
        root_fitness = compute_fitness([self.start_cube.state])[0][0]
        root = Node(0, [], self.start_cube.state, root_fitness, self.start_cube.is_solved(), 0)
        
        if root.solved:
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
                tempcube.from_state(node.cube)
                tempcube.move(move)
            
                if tempcube.is_solved():
                    return [Node(0, new_moves, tempcube.state, 0, True, 0)], nodes_searched + 1, True

                if tempcube.__hash__() in seen:
                    continue

                batch_states.append(tempcube.state)
                batch_info.append((tempcube.state, new_moves))
                seen.add(tempcube.__hash__())
                nodes_searched += 1
                
                del tempcube
                
        fitness_scores = compute_fitness(batch_states)
        for (cube_str, new_moves), fitness in zip(batch_info, fitness_scores):
            updated_fitness = fitness[0]
            new_generation.append((Node(0, new_moves, cube_str, updated_fitness, False, 0)))

        new_generation.sort(key=lambda x: x.f)
        return new_generation, nodes_searched, False
    
    def __str__(self) -> str:
        return f"BeamSearch(beam_width={self.beam_width}, max_depth={self.max_depth}, adaptive={self.adaptive})"

def test(search_algos : list, test_file = None):
    from scramble100 import selected_scrambles
    # setting up the initial values
    vals = {}
    for algo in search_algos:
        vals[str(algo)] = {"success": 0, "total_sol_length": 0, "total_nodes": 0, "total_time": 0, "total_error": 0}
        
    if test_file is not None:
        with open(test_file, "a") as f:
            for i, scramble in enumerate(selected_scrambles):
                print(f"Test {i + 1}")
                f.write(f"Test {i + 1}\n")
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

                        # if algo is AWAStar or MAWAStar:
                        if "error" in result:
                            vals[str(algo)]["total_error"] += result["error"]
                            print(f"{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.4f}\t{cube.is_solved()}\t{result['error']}")
                            f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.4f}\t{cube.is_solved()}\t{result['error']}\n")
                        else:
                            print(f"{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.4f}\t{cube.is_solved()}")
                            f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.4f}\t{cube.is_solved()}\n")
                        
                    else:
                        print(f"{vals[str(algo)]['success']}\t{result['success']}")
                        f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{result['success']}\n")
                        
                f.write("\n")
                print()

            f.write("Results:\n")
            print("Results:")
            
            # calculate the average values
            for algo in search_algos:
                if vals[str(algo)]["success"] > 0:
                    vals[str(algo)]["avg_sol_length"] = vals[str(algo)]["total_sol_length"] / vals[str(algo)]["success"]
                    vals[str(algo)]["avg_nodes"] = vals[str(algo)]["total_nodes"] / vals[str(algo)]["success"]
                    vals[str(algo)]["avg_time"] = vals[str(algo)]["total_time"] / vals[str(algo)]["success"]
                    if vals[str(algo)]["total_error"] > 0:
                        vals[str(algo)]["avg_error"] = vals[str(algo)]["total_error"] / vals[str(algo)]["success"]
                else:
                    vals[str(algo)]["avg_sol_length"] = 0
                    vals[str(algo)]["avg_nodes"] = 0
                    vals[str(algo)]["avg_time"] = 0
                    
                if "avg_error" in vals[str(algo)]:
                    f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.4f}\t{vals[str(algo)]['avg_time']:.4f}\t{vals[str(algo)]['avg_error']:.4f}\n")
                    print(f"{str(algo)}\t{vals[str(algo)]['success']}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.4f}\t{vals[str(algo)]['avg_time']:.4f}\t{vals[str(algo)]['avg_error']:.4f}")
                else:
                    f.write(f"{str(algo)}\t{vals[str(algo)]['success'] / 100}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.4f}\t{vals[str(algo)]['avg_time']:.4f}\n")
                    print(f"{str(algo)}\t{vals[str(algo)]['success'] / 100}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.4f}\t{vals[str(algo)]['avg_time']:.4f}")
    else:
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
                    
                    print(f"{vals[str(algo)]['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.4f}\t{cube.is_solved()}")
                else:
                    print(f"{vals[str(algo)]['success']}\tFailed")
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
                
            print(f"{str(algo)}\t{vals[str(algo)]['success'] / 100}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.4f}\t{vals[str(algo)]['avg_time']:.4f}")

if __name__ == "__main__":
    test_file = "results/Beam_Search/results.txt"
    
    search_list = [
        # BeamSearch(beam_width=1000, max_depth=26, adaptive=False),
        # BeamSearch(beam_width=1500, max_depth=26, adaptive=False),
        # BeamSearch(beam_width=2000, max_depth=26, adaptive=False),
        # BeamSearch(beam_width=2500, max_depth=26, adaptive=False),
        # BeamSearch(beam_width=3000, max_depth=26, adaptive=False),
        
        # MBS(beam_width=1000, max_depth=26, adaptive=False),
        # MBS(beam_width=1500, max_depth=26, adaptive=False),
        # MBS(beam_width=2000, max_depth=26, adaptive=False),
        # MBS(beam_width=2500, max_depth=26, adaptive=False),
        # MBS(beam_width=3000, max_depth=26, adaptive=False),
        # MBS(beam_width=4000, max_depth=26, adaptive=False),
        # MBS(beam_width=5000, max_depth=26, adaptive=False),
        # MBS(beam_width=7000, max_depth=26, adaptive=False),
        # MBS(beam_width=10000, max_depth=26, adaptive=False),
        
        # WAStar(scale_factor=1.8, batch_size=1500),
        # WAStar(scale_factor=2.0, batch_size=1500),
        # WAStar(scale_factor=2.2, batch_size=1500),
        # WAStar(scale_factor=2.4, batch_size=1500),
        # WAStar(scale_factor=2.6, batch_size=1500),
        # WAStar(scale_factor=2.8, batch_size=1500),
        # WAStar(scale_factor=3.0, batch_size=1500),
        # WAStar(scale_factor=3.5, batch_size=1500),
        # WAStar(scale_factor=4.0, batch_size=1500),
        # WAStar(scale_factor=5.0, batch_size=1500),
        # WAStar(scale_factor=8.0, batch_size=1500),

        # MWAStar(scale_factor=1.6, batch_size=2000),
        # MWAStar(scale_factor=1.8, batch_size=2000),
        # MWAStar(scale_factor=2.0, batch_size=2000),
        # MWAStar(scale_factor=2.2, batch_size=2000),
        # MWAStar(scale_factor=2.4, batch_size=2000),
        # MWAStar(scale_factor=2.6, batch_size=2000),
        # MWAStar(scale_factor=2.8, batch_size=2000),
        # MWAStar(scale_factor=3.0, batch_size=2000),
        # MWAStar(scale_factor=3.5, batch_size=2000),
        # MWAStar(scale_factor=4.0, batch_size=2000),
        # MWAStar(scale_factor=5.0, batch_size=2000),
        # MWAStar(scale_factor=8.0, batch_size=2000),
        
        # AWAStar(scale_factor=2.0, batch_size=2500, max_time=60),
        # AWAStar(scale_factor=2.5, batch_size=2500, max_time=60),
        # AWAStar(scale_factor=3.0, batch_size=2500, max_time=60),
        # AWAStar(scale_factor=3.5, batch_size=2500, max_time=60),
        # AWAStar(scale_factor=4.0, batch_size=2500, max_time=60),
        # AWAStar(scale_factor=4.5, batch_size=2500, max_time=60),
        # AWAStar(scale_factor=5.0, batch_size=2500, max_time=60),
        
        MAWAStar(scale_factor=1.5, batch_size=3000, max_time=60),
        MAWAStar(scale_factor=2.0, batch_size=3000, max_time=60),
        MAWAStar(scale_factor=2.5, batch_size=3000, max_time=60),
        MAWAStar(scale_factor=3.0, batch_size=3000, max_time=60),
        MAWAStar(scale_factor=3.5, batch_size=3000, max_time=60),
        MAWAStar(scale_factor=4.0, batch_size=3000, max_time=60),
    ]

    test(search_list, test_file)