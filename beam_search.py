import time
import torch
from cube import Cube, load_model, get_allowed_moves, device
import numpy as np
from queue import PriorityQueue

nnet = load_model()

class Beam_Node:
    def __init__(self, cube : Cube, moves, fitness, is_solved):
        self.cube = cube
        self.moves = moves
        self.fitness = fitness
        self.is_solved = is_solved
        
    def __lt__(self, other):
        return self.fitness < other.fitness
    
    def get_possible_moves(self):
        return get_allowed_moves(self.moves)
        
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
        allowed_moves = node.get_possible_moves()
        
        for move in allowed_moves:            
            new_moves = node.moves + [move]
            tempcube = node.cube.copy()
            tempcube.move(move)
            
            if tempcube in seen_state:
                continue
        
            if tempcube.is_solved():
                return [Beam_Node(tempcube, new_moves, 0, True)], nodes_searched, True

            batch_states.append(tempcube.state)
            batch_info.append((tempcube, new_moves))
            nodes_searched += 1

    fitness_scores = compute_fitness(batch_states)
    for (tempcube, new_moves), fitness in zip(batch_info, fitness_scores):
        new_generation.append(Beam_Node(tempcube, new_moves, fitness, False))

    # Sort new generation based on fitness
    new_generation.sort(key=lambda x: x.fitness)
    return new_generation, nodes_searched, False

def beam_search(scrambled_cube : Cube, beam_width = 1024, max_depth = 100, prevention = True) -> dict:
    root_fitness = compute_fitness([scrambled_cube.state])[0]
    root = Beam_Node(scrambled_cube, [], root_fitness, scrambled_cube.is_solved())
    generation = [root]
    start_time = time.time()
    node_searched = 1
    
    seen_state = set()
    seen_state.add(scrambled_cube)
    
    for depth in range(max_depth + 1):
              
        if depth == max_depth:
            return {"success" : False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}
                
        new_generation, searched_nodes, success = generate_new_generation(generation, prevention)
        
        if success:
            return {"success" : True, "solutions": new_generation[0].moves, "num_nodes": node_searched + searched_nodes, "time_taken": time.time() - start_time}
        
        node_searched += searched_nodes
        generation = new_generation[: int(beam_width * (1 + (2 * depth)/100))]
        
        for node in generation:
            seen_state.add(node.cube)

    raise Exception("Beam search failed to find a solution")

def generate_new_generation_pq(generation: PriorityQueue, seen_state):
    new_generation = PriorityQueue()
    nodes_searched = 0
    
    batch_states = []
    batch_info = []
    
    while not generation.empty():
        _, node = generation.get()
        allowed_moves = node.get_possible_moves()
        
        for move in allowed_moves:
            new_moves = node.moves + [move]
            tempcube = node.cube.copy()
            tempcube.move(move)
            
            if tempcube in seen_state:
                continue
        
            if tempcube.is_solved():
                ans_queue = PriorityQueue()
                ans_queue.put((0, Beam_Node(tempcube, new_moves, 0, True)))
                return ans_queue, nodes_searched, True

            batch_states.append(tempcube.state)
            batch_info.append((tempcube, new_moves))
            nodes_searched += 1

    fitness_scores = compute_fitness(batch_states)
    for (tempcube, new_moves), fitness in zip(batch_info, fitness_scores):
        new_generation.put((fitness, Beam_Node(tempcube, new_moves, fitness, False)))

    return new_generation, nodes_searched, False


def beam_search_pq(scrambled_cube: Cube, beam_width=1024, max_depth=100) -> dict:
    root_fitness = compute_fitness([scrambled_cube.state])[0]
    root = Beam_Node(scrambled_cube, [], root_fitness, scrambled_cube.is_solved())
    generation = PriorityQueue()
    generation.put((root_fitness, root))
    start_time = time.time()
    node_searched = 1
    
    seen_state = set()
    seen_state.add(scrambled_cube)

    for depth in range(max_depth + 1):
        
        print(f"Depth: {depth}, min_fitness: {generation.queue[0][0]}, best_moves: {generation.queue[0][1].moves}, num_nodes: {node_searched}")
        
        if depth == max_depth:
            return {"success": False, "solutions": None, "num_nodes": node_searched, "time_taken": time.time() - start_time}
        
        new_generation, searched_nodes, success = generate_new_generation_pq(generation, seen_state)
        
        if success:
            solution_node = new_generation.get()[1]
            return {"success": True, "solutions": solution_node.moves, "num_nodes": node_searched + searched_nodes, "time_taken": time.time() - start_time}
        
        node_searched += searched_nodes
        generation = PriorityQueue()
        
        for _ in range(min(int(beam_width * (1 + (2 * depth)/100)), new_generation.qsize())):
            if new_generation.empty(): 
                break
            
            fitness, node = new_generation.get()
            generation.put((fitness, node))
            seen_state.add(node.cube)

    raise Exception("Beam search failed to find a solution")
            
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
    """
    
    for i, scramble in enumerate(selected_scrambles):
        print(f"Test {i + 1}")
        cube = Cube()
        cube.move_list(cube.convert_move(scramble))
        
        result = beam_search_pq(cube, 1911, 35)
        if result["success"]:
            success += 1
            total_sol_length += len(result["solutions"])
            total_nodes += result["num_nodes"]
            total_time += result["time_taken"]
            cube.move_list(result["solutions"])
            print(f"Success: {success}, Sol Length: {len(result['solutions'])}, Num Nodes: {result['num_nodes']}, Time Taken: {result['time_taken']}, Check: {cube.is_solved()}")
    
    print(f"Success Rate: {success/100}, Avg Sol Length: {total_sol_length/success}, Avg Num Nodes: {total_nodes/success}, Avg Time Taken: {total_time/success}")