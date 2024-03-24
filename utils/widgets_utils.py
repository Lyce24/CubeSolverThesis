from dataclasses import dataclass
import numpy as np
import utils.cube_utils as cube

"""
Colors:

             0 1 2
             3 4 5
             6 7 8
    9 10 11  18 19 20  27 28 29  36 37 38 
    12 13 14 21 22 23  30 31 32  39 40 41
    15 16 17 24 25 26  33 34 35  42 43 44
            45 46 47
            48 49 50
            51 52 53
            
translate to:
                      |  2  5  8 |
                      |  1  4  7 |
                      |  0  3  6 |
             --------------------------------------------
             20 23 26 | 47 50 53 | 29 32 35 | 38 41 44
             19 22 25 | 46 49 52 | 28 31 34 | 37 40 43
             18 21 24 | 45 48 51 | 27 30 33 | 36 39 42
             --------------------------------------------           
                      | 11 14 17 |
                      | 10 13 16 |
                      | 9  12 15 |
"""

cubie_to_state = {
    0 : 2, 1 : 5, 2 : 8, 3 : 1, 4 : 4, 5 : 7, 6 : 0, 7 : 3, 8 : 6,
    9 : 20, 10 : 23, 11 : 26, 12 : 19, 13 : 22, 14 : 25, 15 : 18, 16 : 21, 17 : 24,
    18 : 47, 19 : 50, 20 : 53, 21 : 46, 22 : 49, 23 : 52, 24 : 45, 25 : 48, 26 : 51,
    27 : 29, 28 : 32, 29 : 35, 30 : 28, 31 : 31, 32 : 34, 33 : 27, 34 : 30, 35 : 33,
    36 : 38, 37 : 41, 38 : 44, 39 : 37, 40 : 40, 41 : 43, 42 : 36, 43 : 39, 44 : 42,
    45 : 11, 46 : 14, 47 : 17, 48 : 10, 49 : 13, 50 : 16, 51 : 9, 52 : 12, 53 : 15
}

state_to_cubie = {v: k for k, v in cubie_to_state.items()}

color_dict = {
    "blue": 0,
    "green" : 1,
    "orange" : 2,
    "red" : 3,
    "yellow" : 4,
    "white" : 5
}


COLOR_VALUE = {"blue":0,"orange":1,"white":2,"red":3,"yellow":4,"green":5}
VALUE_COLOR = ["blue","orange","white","red","yellow","green"]
ACTIONS = ["U", "L", "F", "R", "B", "D", "U'", "L'", "F'", "R'", "B'", "D'"]


def seq_to_string(seq):
    global ACTIONS
    string_seq = ""
    for action in seq:
        string_seq = string_seq+" "+ACTIONS[action]
    return string_seq

def convert_to_res_state(cubie):
    state = np.array([0]*54)
    for i, cub in enumerate(cubie):
        state[cubie_to_state[i]] = color_dict[cub[1]]
    return state

def cubies_to_state(cubies):
    global COLOR_VALUE
    state = np.zeros((54,),dtype=np.ushort)
    for idx,cubie in enumerate(cubies):
        state[idx] = COLOR_VALUE[cubie[1]]
    return state

def scramble(cubies,seq=None):
    global VALUE_COLOR
    
    state = cubies_to_state(cubies)
    states = []
    
    for action in seq:
        state = state[cube.idxs[action]]
        states.append(state.copy())
        
    return states,seq

@dataclass()
class Settings:
    search_weight: float = 0.3
    search_batch_size: int = 3000
    
    search_beam_width: int = 2000
    adaptive_beam_search: bool = False
    beam_search_max_depth: int = 100
    
    actions_delay: int = 250
    def_scramble_depth: int = 20
    def_time: int = 30
    def_iter: int = 100


    
if __name__=="__main__":
    str = "B F U R U F' F B' U' D' F U' I "
    
    cubie = [['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'blue'], ['things', 'white'], ['things', 'white'], ['things', 'white'], ['things', 'orange'], ['things', 'orange'], ['things', 'orange'], ['things', 'orange'], ['things', 'orange'], ['things', 'orange'], ['things', 'red'], ['things', 'red'], ['things', 'red'], ['things', 'white'], ['things', 'white'], ['things', 'white'], ['things', 'white'], ['things', 'white'], ['things', 'white'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'red'], ['things', 'red'], ['things', 'red'], ['things', 'red'], ['things', 'red'], ['things', 'red'], ['things', 'orange'], ['things', 'orange'], ['things', 'orange'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'yellow'], ['things', 'green'], ['things', 'green'], ['things', 'green'], ['things', 'green'], ['things', 'green'], ['things', 'green'], ['things', 'green'], ['things', 'green'], ['things', 'green']]
    
    print(convert_to_res_state(cubie))