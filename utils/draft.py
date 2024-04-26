from cube import Cube, load_model, device, get_allowed_moves, Move
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
        
        
class RWAStar:
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
    
    
from cube import Cube, inverse_moves, move_dict

scramble_list_test = [
  "U R L D R' B U2 R B' U F' D' B' D2 R' L' B' F' U' R L U B U R D R B2 U B D2 R L' U2 B2 U L' F' R2 L F2 R U2 B D R' F R L U2 B' R' B U L2 F2 R U R2 L D2 U2 F2 R' L' D2 U B2 F U2 R' U2 R' L2 D L' D2 B2 U' B2 U2 R' D L' D' R' F D F R2 L2 D2 B' U2 B D U' B R2 U",
  "U R' B' F' R' D2 B2 D2 U B2 F D U2 F2 D' L2 U2 F' L2 F U B2 D R2 D2 U2 R2 L2 D R2 D' L' B2 D2 F2 L2 D F2 R2 U' F2 D2 F' D' B L' D F U' B U' F2 L' B' L2 D' R L B F U' L' B D B R2 D B' U' L F' R2 L2 D' F2 D U' L' B R2 F R U' F D U2 L B2 U2 F' L F' U' F' L2 U F R2 U2 B'",
  "B2 R B' D2 F R' B' F' U2 L B U B L' U2 L B2 R' F' L2 B2 L2 B2 U L2 U' B' R L B' F2 L' U' F' U B' U' F' D' R F R' U' L' U2 F' R U R' L' F2 R2 F' L2 U B L' B F' D L2 U B F U B' D2 F2 U B U' L' U L2 B2 R' U2 R2 D2 B2 D' R U2 F2 D R' F2 L2 B L' B2 U' R2 D L2 F2 D2 F2 L' D'",
  "F' R L' D' L' F D' B' F' D' F L' B U' B' D F2 D' L' F U' L2 D' B2 U B R' U2 L' B2 D F' L D F2 U2 B2 U' R' F' L B' U' L' U2 R D2 L F U2 L2 U' L F2 U' L' F2 R' D2 F2 L' D L2 U2 R D2 U F' U2 R2 L U2 L D U B F R2 U2 R F' L B U' L D U2 L D' U2 L' B' L' U2 L2 U F2 R F' D'",
  "L2 F2 L F D2 F2 U2 R' D F U L' F' R2 F2 L F L2 U2 L U' B U2 L U2 L D R2 L' D' F2 D2 L2 B F' U2 R' F U2 F D F' D' R' B2 U' L2 D' U F D2 F2 D B F2 D2 L' U B2 F2 L' B2 L' D' L2 U B' R L D2 L2 B' U2 R F L D L' F2 D L F R' F D2 F' U2 R U' L2 F' U2 R' U' B2 L2 U R' B2 F'",
  "B L' D R' F R' L' B' R2 U2 B' U R L2 D2 F2 L' F' D F2 D U2 F2 D' L2 B2 L' F L U B U2 R' L2 F' R2 U' B' D L2 D R' D R U R B2 D' L F D' U F' L2 F2 U2 R' B' F' U' F' U L' D2 F2 L' F2 R' B2 F L2 D' L' B2 D2 R2 D' L' B' L2 D2 U' B2 D' B F U2 R' F U2 F2 U2 F L' D2 F L' F2 D F'",
  "F2 D2 B2 L2 D' B2 L2 U2 B' L' U2 B2 U F R U' F D2 L' F' L' B D F2 R2 U2 R2 D' U' L F U B2 R2 L' B D U' L2 D F2 L2 D U F' D2 L2 U L F D' F' D2 B U' L2 F2 R' U2 R2 B U' B' D R' B2 F' U' B' U L2 F' L B' D2 B2 R2 D2 R2 F U' R' L2 D' F2 U B' D B R' F' U B' D2 U' L2 U2 L' D L'",
  "L' U B D' L' F' D2 F2 R2 U' L' F' D R' L F2 D R2 D B U' B2 D2 R D2 F U2 L' B2 U2 R' L2 U2 B2 L' B2 D' U' F2 U' F' D R B' L' B2 L2 D' U L B' D U2 F2 L2 B D2 L' D2 U' L' F' U' F2 D L2 B2 R' F' U R D B' F' U' L' F2 U F2 D' F2 L2 B' U' L U L' B D2 L' D R' U2 F2 D' U' L U R2 L'",
  "F' D U2 R' L B2 U2 L' B' D' L2 D R D F R L B' U2 L D F R' B' D2 R B2 L' U' F2 U2 F2 U' B R D B' R' U F L' U L2 B2 U' F L' D' U' F' D2 U2 L' B L' F' U L F2 R D' L U2 R2 F' L' U F' L U2 R2 D' B2 L D' R D L U F L B' D F2 L2 B L U L2 U2 B2 R L2 U' F' L' F2 R' B' U2",
  "B F' L' F2 D B R' U2 L2 U' F' L2 B' R' U B' U2 F2 U F2 L U2 B' R' F2 D' B2 R2 L2 B2 D' B U2 R2 L' B2 U F2 R2 D' L B2 R' U2 R' L U2 B' L2 D' L' F L B U F' L' F D' U L2 U' B2 R' D F R2 U F2 U2 F D2 L2 U R2 U2 F2 U F' D B2 F2 R U R' F D' U L F R D U' L2 F2 D' F2 L B2 L"
]

f01 = 'FLLUUBUUDLDUFRRBRLRFFUFDRLUDBRLDBRDBLBFFLLDFFBDUUBRDRB'
f02 = 'BRBBUFFLLFLDBRDFLFRDDFFRBFRDDULDBBUURDDULULURLFURBRLBU'
f03 = 'DBLRUULUUFFBLRDFLBBRLUFFDFRFDUDDUBBRLFURLBUBRDRFLBDDLR'
f04 = 'UFBDULRLULDDRRRFDUBBBLFFLFLUUDRDFRBRRBDRLUFBFLLFUBDBUD'
f05 = 'BLURULUBBDFRRRBDDDRDRRFFFURLFFUDLDDLUUFLLDFBUBBLRBUBFL'
f06 = 'FRRRUFDBFLLBURBRBDLUUUFRDDFBRDDDDLDBUBFLLFULLUFRLBURFB'
f07 = 'FBBDUFRDDBUURRRLFUDBLRFBBLUDFBUDRDULUFFDLUFLRRLRDBLFBL'
f08 = 'UDFDULRFRBUUDRRLLLFRDRFBFRBDBDBDFDUUBFULLDRULLLRUBBBFF'
f09 = 'BDFBURLUDFBLRRUBDFDLRRFDRRRUFDFDLLBRULBULUFLBDFLFBBUDU'
f10 = 'RUUFUBFRRFUBBRRBLLRFULFDFDRLFDLDDFRBUUDULBLFULLBBBRDDD'

scramble_str_list = [f01, f02, f03, f04, f05, f06, f07, f08, f09, f10]

solution_list = [
    "R B U U L L U F F R R F F U F' U U R R F' U' F F L B L'",
    "B' L L R' F U' B' R F' D D B U U D R U F B' L B U U",
    "U U B' U F B U' R F' R R F' U' R B' L D' L L R U U",
    "R F U' D' B B R B B D D F' R' B U U F R D D R R U U D'",
    "U U B L R' F F U U L' F F R U' L L F' L R' U L' R R D'",
    "B L U' L' U' B R U' B' U' L L U' F R F B R U",
    "D' F F U' R R U L U U B' L' R' U U B' R' F F U' F B B R'",
    "L' R R D L' R R B' D' F' R B R' U U D' B B L' F F R'",
    "D' B B L F F R' B B L D D F F U F' U' R F' D L' U L L",
    "F D D F' B D F L F U' F F L L B' L B R' F F L' R'",
    "R U U L U U L L B B R R D L' B B U F' B D R' L F' U",
    "U R' B' R R F' R R L' D D B D F F L' B' D' L F F D' L L",
    "U D D B L L B' L L B B R' D F' U' R' D L' F' L F' B'",
    "R F R R L L U B D B B L F' U U L' F F L D F F U'",
    "L D L L U U F U D D F F U R F B R R U' R F U' F'",
    "U F F U' D F F L B' D D F R D' L L D F F D B' U' F'",
    "U D D R' L' B B R R D L D F' U U D D R B U D F F D D",
    "L' B B D B B U L' U L L U R U U B D L' F F R F' R'",
    "L B B U D R F B D D L' U U D F' U U F' U R B B",
    "U' R R B B U U B' D' L D L L U U D D R' U D R' D B' U"
]


scramble_list = [
    "L B' L' F' F' U F R' R' U' U' F U' F' F' R' R' F' F' U' L' L' U' U' B' R'",
    "U' U' B' L' B F' U' R' D' U' U' B' D' D' F R' B U F' R L' L' B",
    "U' U' R' L' L' D L' B R' U F R' R' F R' U B' F' U' B U' U'",
    "D U' U' R' R' D' D' R' F' U' U' B' R F D' D' B' B' R' B' B' D U F' R'",
    "D R' R' L U' R L' F L' L' U R' F' F' L U' U' F' F' R L' B' U' U'",
    "U' R' B' F' R' F' U L' L' U B U R' B' U L U L' B'",
    "R B' B' F' U F' F' R B U' U' R L B U' U' L' U' R' R' U F' F' D",
    "R F' F' L B' B' D U' U' R B' R' F D B R' R' L D' R' R' L",
    "L' L' U' L D' F R' U F U' F' F' D' D' L' B' B' R F' F' L' B' B' D",
    "R L F' F' R B' L' B L' L' F' F' U F' L' F' D' B' F D' D' F'",
    "U' F L' R D' B' F U' B' B' L D' R' R' B' B' L' L' U' U' L' U' U' R'",
    "L' L' D F' F' L' D B L F' F' D' B' D' D' L R' R' F R' R' B R U'",
    "B F L' F L D' R U F D' R B' B' L' L' B L' L' B' D' D' U'",
    "U F' F' D' L' F' F' L U' U' F L' B' B' D' B' U' L' L' R' R' F' R'",
    "F U F' R' U R' R' B' F' R' U' F' F' D' D' U' F' U' U' L' L' D' L'",
    "F U B D' F' F' D' L' L' D R' F' D' D' B L' F' F' D' U F' F' U'",
    "D' D' F' F' D' U' B' R' D' D' U' U' F D' L' D' R' R' B' B' L R D' D' U'",
    "R F R' F' F' L D' B' U' U' R' U' L' L' U' L U' B' B' D' B' B' L",
    "B' B' R' U' F U' U' F D' U' U' L D' D' B' F' R' D' U' B' B' L'",
    "U' B D' R D' U' R D' D' U' U' L' L' D' L' D B U' U' B' B' R' R' U"
]

validation_states = [
    [0, 1, 0, 2, 0, 0, 4, 5, 2, 0, 1, 3, 2, 1, 0, 3, 2, 1, 4, 3, 3, 5, 2, 3, 0, 2, 2, 4, 1, 2, 0, 3, 3, 1, 5, 1, 5, 0, 5, 5, 4, 3, 2, 4, 5, 4, 4, 5, 4, 5, 1, 3, 4, 1],
    [2, 3, 1, 5, 0, 2, 4, 3, 0, 3, 0, 0, 2, 1, 3, 0, 1, 4, 5, 4, 4, 4, 2, 0, 3, 2, 1, 3, 0, 3, 5, 3, 5, 5, 4, 4, 2, 1, 2, 5, 4, 1, 1, 3, 2, 5, 4, 5, 1, 5, 0, 1, 2, 0],
    [3, 2, 0, 0, 0, 0, 3, 0, 4, 2, 5, 2, 2, 1, 3, 0, 1, 3, 5, 4, 2, 2, 2, 0, 5, 5, 4, 1, 5, 1, 4, 3, 5, 3, 1, 1, 5, 3, 2, 1, 4, 3, 0, 2, 4, 1, 3, 0, 4, 5, 4, 4, 1, 5],
    [0, 0, 0, 1, 0, 5, 3, 0, 2, 5, 3, 0, 5, 1, 2, 0, 5, 5, 2, 2, 4, 4, 2, 4, 2, 2, 5, 1, 4, 4, 1, 3, 3, 5, 1, 4, 2, 4, 1, 0, 4, 3, 1, 0, 3, 4, 1, 3, 5, 5, 3, 3, 2, 1],
    [5, 2, 5, 5, 0, 4, 1, 0, 4, 1, 0, 5, 1, 1, 1, 4, 5, 3, 5, 1, 3, 2, 2, 5, 0, 5, 0, 4, 4, 3, 3, 3, 4, 0, 3, 1, 2, 0, 2, 2, 4, 3, 2, 3, 1, 3, 1, 2, 4, 5, 0, 0, 2, 4],
    [3, 0, 2, 1, 0, 3, 3, 2, 1, 0, 3, 0, 3, 1, 0, 4, 5, 5, 2, 1, 1, 4, 2, 4, 4, 1, 0, 2, 5, 1, 2, 3, 4, 0, 0, 4, 2, 5, 3, 0, 4, 1, 5, 5, 4, 3, 4, 5, 2, 5, 2, 1, 3, 5],
    [5, 5, 2, 5, 0, 4, 1, 2, 3, 0, 2, 4, 1, 1, 5, 4, 4, 0, 3, 3, 5, 1, 2, 0, 0, 3, 0, 5, 1, 5, 0, 3, 0, 2, 2, 1, 1, 4, 4, 3, 4, 1, 4, 4, 1, 2, 0, 2, 2, 5, 3, 3, 5, 3],
    [4, 2, 5, 2, 0, 2, 0, 4, 3, 1, 5, 0, 3, 1, 5, 2, 2, 4, 5, 4, 0, 3, 2, 4, 4, 3, 1, 3, 1, 2, 5, 3, 3, 0, 0, 5, 5, 5, 1, 1, 4, 1, 2, 0, 3, 3, 0, 2, 1, 5, 0, 1, 4, 4],
    [2, 1, 5, 4, 0, 0, 3, 5, 4, 0, 4, 4, 5, 1, 1, 4, 5, 3, 4, 3, 2, 0, 2, 5, 3, 2, 1, 0, 1, 1, 0, 3, 3, 2, 4, 3, 1, 3, 1, 2, 4, 3, 2, 1, 0, 0, 0, 5, 4, 5, 2, 5, 2, 5],
    [3, 2, 1, 1, 0, 5, 3, 0, 0, 2, 2, 3, 3, 1, 0, 2, 3, 5, 0, 5, 4, 4, 2, 0, 1, 2, 1, 3, 3, 0, 1, 3, 4, 5, 4, 2, 1, 1, 5, 4, 4, 0, 4, 2, 2, 5, 1, 4, 3, 5, 5, 0, 5, 4],
    [1, 3, 0, 5, 0, 3, 1, 3, 1, 3, 4, 0, 5, 1, 1, 4, 2, 4, 0, 0, 5, 0, 2, 1, 2, 0, 5, 1, 2, 3, 4, 3, 5, 3, 1, 3, 0, 2, 5, 0, 4, 4, 5, 2, 2, 4, 3, 2, 4, 5, 1, 2, 5, 4],
    [4, 5, 5, 4, 0, 5, 1, 0, 1, 5, 4, 0, 1, 1, 3, 2, 5, 1, 1, 3, 0, 2, 2, 2, 2, 1, 0, 2, 2, 3, 0, 3, 4, 0, 3, 5, 5, 0, 2, 2, 4, 3, 3, 1, 3, 4, 5, 3, 4, 5, 1, 4, 0, 4],
    [1, 4, 4, 2, 0, 2, 5, 1, 1, 0, 0, 2, 4, 1, 0, 5, 4, 5, 4, 1, 3, 3, 2, 3, 5, 0, 2, 0, 3, 3, 2, 3, 5, 0, 0, 4, 3, 4, 3, 1, 4, 5, 2, 3, 0, 1, 5, 4, 2, 5, 1, 2, 5, 1],
    [0, 3, 5, 1, 0, 0, 3, 2, 5, 5, 3, 1, 4, 1, 4, 1, 4, 0, 2, 2, 2, 4, 2, 0, 5, 5, 3, 4, 3, 4, 1, 3, 0, 4, 5, 3, 2, 3, 0, 0, 4, 5, 1, 5, 0, 3, 1, 4, 2, 5, 2, 2, 1, 1],
    [0, 4, 1, 3, 0, 4, 0, 5, 1, 3, 4, 1, 0, 1, 1, 0, 0, 3, 0, 1, 4, 2, 2, 1, 3, 5, 2, 5, 4, 4, 5, 3, 2, 2, 2, 5, 4, 0, 2, 3, 4, 3, 5, 5, 2, 4, 3, 5, 2, 5, 1, 1, 0, 3],
    [3, 3, 3, 1, 0, 4, 2, 3, 4, 5, 4, 0, 4, 1, 4, 5, 0, 5, 3, 3, 4, 0, 2, 1, 4, 1, 5, 0, 2, 1, 5, 3, 5, 1, 0, 2, 2, 2, 0, 3, 4, 2, 0, 0, 1, 3, 5, 1, 1, 5, 2, 2, 5, 4],
    [1, 3, 0, 4, 0, 1, 4, 3, 2, 5, 3, 3, 1, 1, 0, 0, 4, 2, 2, 5, 4, 0, 2, 1, 1, 3, 2, 5, 4, 1, 1, 3, 5, 3, 2, 0, 5, 5, 4, 2, 4, 5, 1, 0, 3, 5, 4, 4, 2, 5, 2, 0, 0, 3],
    [5, 3, 4, 0, 0, 3, 4, 2, 5, 2, 4, 1, 3, 1, 5, 5, 0, 2, 5, 0, 3, 0, 2, 1, 3, 1, 3, 4, 2, 0, 3, 3, 5, 3, 4, 2, 1, 1, 1, 4, 4, 5, 0, 5, 0, 4, 2, 0, 1, 5, 2, 1, 4, 2],
    [4, 5, 5, 4, 0, 0, 1, 4, 4, 5, 0, 5, 5, 1, 2, 0, 3, 3, 2, 0, 3, 3, 2, 2, 2, 1, 2, 4, 5, 3, 1, 3, 2, 3, 5, 1, 5, 3, 2, 1, 4, 4, 1, 2, 1, 0, 4, 0, 1, 5, 3, 0, 0, 4],
    [2, 2, 4, 1, 0, 3, 3, 5, 2, 3, 2, 1, 5, 1, 5, 3, 3, 2, 5, 4, 3, 4, 2, 0, 4, 5, 5, 1, 0, 4, 0, 3, 0, 5, 2, 0, 0, 1, 4, 3, 4, 4, 1, 1, 0, 2, 1, 0, 2, 5, 3, 5, 4, 1]
]


if __name__ == "__main__":
    cube = Cube()
    for i, scramble in enumerate(scramble_list):
        cube.move_list(cube.convert_move(scramble))
        
        cube.move_list(cube.convert_move(solution_list[i]))
        print(cube.is_solved())
        cube.reset()
        
    # find the max length of the scramble
    max_length = 100
    for scramble in scramble_list:
        list_move = cube.convert_move(scramble)
        max_length = min(max_length, len(list_move))
        
    print(max_length)
        

  
    from scramble100 import selected_scrambles
    
    test_str = selected_scrambles[0]
    
    temp = Cube()
    temp.move_list(temp.convert_move(test_str))
    
    awa = RWAStar(temp, 5, 1)
    
    result, error = awa.search()
    print(result, error)
    
    