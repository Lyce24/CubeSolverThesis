from cube import cube
import torch
from networks.getNetwork import getNetwork
from widgets import widgets_utils
from search.search import AStar
from cube.cube import clear_seq
from configs.solve_conf import SolveConfig
import argparse
import sys
import time

file = "test100.txt"

class Solve_Cube(object):
    def __init__(self, conf):
        self.conf = conf
        self.net = getNetwork(conf.network_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.net(54 * 6).to(device)
        self.net.load_state_dict(torch.load(conf.heuristic_path, map_location=device))

        self.astar = AStar(weight=conf.weight, max_iter=None, max_time=None, batch_size=conf.batch_size)

        self.COLOR_VALUE = {"blue": 0, "orange": 1, "white": 2, "red": 3, "yellow": 4, "green": 5}
        self.VALUE_COLOR = ["blue", "orange", "white", "red", "yellow", "green"]

        self.cubies = []
        solved = cube.get_solved()
        for _, value in enumerate(solved):
            self.cubies.append(["", self.VALUE_COLOR[value]])
            
            
    def solve(self):

        threshold = int(100)
        self.astar.set_max_time(threshold)
            
        state = widgets_utils.cubies_to_state(self.cubies)
        
        start_time = time.time()
        solved,iteration,nodes,solution = self.astar.search(state,self.net)
        finish_time = time.time()
        total_time = finish_time - start_time
        
        if solved:
            with open(file, "a") as f:
                f.write(f"Scramble: {widgets_utils.seq_to_string(test_cube.seq)}\tIteration: {iteration}\tTime: {total_time}\tNodes: {nodes}\tSolution: {widgets_utils.seq_to_string(solution)}\n")
            print(f"Iteration: {iteration}\tTime: {total_time}\tNodes: {nodes}\tSolution: {widgets_utils.seq_to_string(solution)}")
            return True
        else:
            print("No solution found")
            return False

    def reset(self):
        solved = cube.get_solved()
        for idx, value in enumerate(solved):
            self.cubies[idx][0] = ""
            self.cubies[idx][1] = self.VALUE_COLOR[value]

    def scramble(self):
        depth = 26
        self.states, self.seq = widgets_utils.scramble(self.cubies, depth=depth)
        for idx, value in enumerate(self.states[-1]):
            self.cubies[idx][0] = ""
            self.cubies[idx][1]=self.VALUE_COLOR[value]


if __name__ == "__main__":
    test_cube = Solve_Cube(SolveConfig("ini_files/solve.ini"))
    
    index = 1
    for _ in range(100):
        test_cube.scramble()
        if test_cube.solve():
            index += 1
        test_cube.reset()
        
    print(f"Total solved: {index}")
    print(f"Success Rate: {index/100}")