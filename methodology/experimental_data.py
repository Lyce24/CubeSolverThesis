from search import MWAStar, WAStar, BeamSearch, MBS, MAWAStar, compute_fitness, Node
from cube import Cube, get_allowed_moves
import time
from typing import Union

class BeamSearchWithOutSS:
    def __init__(self, start_cube : Union[Cube, None] = None, beam_width = 1000, max_depth = 40):
        self.start_cube = start_cube
        self.beam_width = beam_width
        self.max_depth = max_depth
        
    def search(self) -> dict:
        
        print(f"Starting Beam Search (without SS) with beam width = {self.beam_width} and max depth = {self.max_depth}")
            
        assert self.start_cube is not None, "Start cube is not set"    
        
        node_searched = 1
        start_time = time.time()
        
        root_fitness = compute_fitness([self.start_cube.state])[0][0]
        root = Node(0, [], self.start_cube.state, root_fitness, self.start_cube.is_solved(), 0)
        
        if root.solved:
            return {"success" : True, "solutions": [], "length": 0, "num_nodes": 1, "time_taken": 0}
        
        generation = [root]

        for depth in range(self.max_depth + 1):
                
            if depth == self.max_depth:
                return {"success" : False}
                    
            new_generation, searched_nodes, success = self.generate_new_generation(generation)
                    
            if success:
                return {"success" : True, "solutions": new_generation[0].moves, "length": len(new_generation[0].moves), "num_nodes": node_searched, "time_taken": time.time() - start_time}
            
            node_searched += searched_nodes
            
            generation = new_generation[: self.beam_width]
                        
        return {"success" : False}

        
    def generate_new_generation(self, generation: list[Node]) -> tuple:
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

                batch_states.append(tempcube.state)
                batch_info.append((tempcube.state, new_moves))
                nodes_searched += 1
                
                del tempcube
                
        fitness_scores = compute_fitness(batch_states)
        for (cube_str, new_moves), fitness in zip(batch_info, fitness_scores):
            updated_fitness = fitness[0]
            new_generation.append((Node(0, new_moves, cube_str, updated_fitness, False, 0)))

        new_generation.sort(key=lambda x: x.f)
        return new_generation, nodes_searched, False
    
    def __str__(self) -> str:
        return f"BeamSearchWithoutSS(beam_width={self.beam_width}, max_depth={self.max_depth})"


def test(search_algos : list, selected_scrambles : list, test_file = None):
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
                            print(f"{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.2f}\t{result['error']}\t{cube.is_solved()}")
                            f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.2f}\t{result['error']}\t{cube.is_solved()}\n")
                        else:
                            print(f"{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.2f}\t{cube.is_solved()}")
                            f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{result['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.2f}\t{cube.is_solved()}\n")
                        
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
                    f.write(f"{str(algo)}\t{vals[str(algo)]['success']}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.2f}\t{vals[str(algo)]['avg_time']:.2f}\t{vals[str(algo)]['avg_error']:.2f}\n")
                    print(f"{str(algo)}\t{vals[str(algo)]['success']}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.2f}\t{vals[str(algo)]['avg_time']:.2f}\t{vals[str(algo)]['avg_error']:.2f}")
                else:
                    f.write(f"{str(algo)}\t{vals[str(algo)]['success'] / len(selected_scrambles)}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.2f}\t{vals[str(algo)]['avg_time']:.2f}\n")
                    print(f"{str(algo)}\t{vals[str(algo)]['success'] / len(selected_scrambles)}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.2f}\t{vals[str(algo)]['avg_time']:.2f}")
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
                    
                    print(f"{vals[str(algo)]['success']}\t{len(result['solutions'])}\t{result['num_nodes']}\t{result['time_taken']:.2f}\t{cube.is_solved()}")
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
                
            print(f"{str(algo)}\t{vals[str(algo)]['success'] / len(selected_scrambles)}\t{vals[str(algo)]['avg_sol_length']}\t{vals[str(algo)]['avg_nodes']:.2f}\t{vals[str(algo)]['avg_time']:.2f}")

if __name__ == "__main__":
    
    
    scrambles = [
        "R R D' R R D D R R L L D' L' B' D D F' D U U B' R U' F B B",
        "B B L L D D R R B B U L L U R' D U U R B B R R D D F D U' R F'",
        "U L L B B L L D' R R U' L L B' L' F' U R D R R B' R R U F U'",
        "U U F F U F F U U R R D D U' L L F D' B B D R B U U B B D' R' U",
        "F F D L L B B U U R R D R R B B D D F' R B B D D L' D D L U R' U U"
    ]


    # weighted_atar_search = WAStar(None, 2.2, 1)
    # batched_weighted_atar_search = WAStar(None, 2.2, len(selected_scrambles)0)
    # mbwa = MWAStar(None, 2.2, 1)
    # mawastar = MAWAStar(None, 2.2, 1)

    algo_list = [
        # MWAStar(scale_factor=2.2, batch_size=500),
        # MWAStar(scale_factor=2.4, batch_size=500),
        # MWAStar(scale_factor=2.6, batch_size=500),
        # MWAStar(scale_factor=2.8, batch_size=500),
        # WAStar(scale_factor=3, batch_size=70),
        # WAStar(scale_factor=3, batch_size=100),
        # WAStar(scale_factor=3, batch_size=200),
        # WAStar(scale_factor=3, batch_size=500),
        # WAStar(scale_factor=3, batch_size=700),
        # WAStar(scale_factor=3, batch_size=1000),
        # WAStar(scale_factor=3, batch_size=2000),
        
        MAWAStar(scale_factor=3, batch_size=70),
        MAWAStar(scale_factor=3, batch_size=100),
        MAWAStar(scale_factor=3, batch_size=200),
        MAWAStar(scale_factor=3, batch_size=500),
        MAWAStar(scale_factor=3, batch_size=700),
        MAWAStar(scale_factor=3, batch_size=1000),
        MAWAStar(scale_factor=3, batch_size=2000),
        
        
        # MAWAStar(scale_factor=3, batch_size=500),
        
        
        
        
        # MWAStar(scale_factor=3.0, batch_size=50),
        # MWAStar(scale_factor=3.0, batch_size=100),
        # MWAStar(scale_factor=3.0, batch_size=200),
        # MWAStar(scale_factor=3.0, batch_size=300),
        # MWAStar(scale_factor=3.0, batch_size=400),
        # MWAStar(scale_factor=3.0, batch_size=500),
        
        # BeamSearchWithOutSS(beam_width=300, max_depth=26),
        # BeamSearchWithOutSS(beam_width=5000, max_depth=26),
        # BeamSearch(beam_width=300, max_depth=26),
        # BeamSearch(beam_width=5000, max_depth=26),
        # MBS(beam_width=300, max_depth=26),
        # MBS(beam_width=5000, max_depth=26),
        
        
        # BeamSearchWithOutSS(beam_width=500, max_depth=26),
        # BeamSearch(beam_width=500, max_depth=26),
        
    ]

    test(algo_list, scrambles)