from rubik54 import Cube
import random
from utils.cube_utils import Move, move_dict



class MCTS_Node:
    def __init__(self, mcts_agent, state, moves):
        
        cube = Cube()
        cube.from_string(state)
        self.state = state
        self.moves = moves
        self.children = []
        self.visits = 0
        self.value = 0
        self.is_terminal = False
        self.is_solved = False
        
        if cube.is_solved():
            self.is_terminal = True
            self.is_solved = True
            self.value = 1
            return
        
        self.mcts_agent = mcts_agent
        self.state = state
        self.children = []
        self.visits = 0
        self.value = 0
        self.is_terminal = False
        self.is_solved = False
        
        
        
    def upper_confidence_bound(self, parent_visits):
        return self.value + self.mcts_agent.exploration_constant * (parent_visits / (1 + self.visits))
            
            
    