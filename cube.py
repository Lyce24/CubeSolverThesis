import random
import numpy as np
import torch
from collections import OrderedDict
import re
from enum import IntEnum
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 6, 5000, 1000, 4, 1,
class ResnetModel(nn.Module):
    def __init__(self, state_dim: int = 54, one_hot_depth: int = 6, h1_dim: int = 5000, resnet_dim: int = 1000, num_resnet_blocks: int = 4,
                 out_dim: int = 1, batch_norm: bool = False):
        
        # state_dim, 6, 5000, 1000, 4, 1, True)
        super().__init__()
        self.one_hot_depth: int = one_hot_depth
        self.state_dim: int = state_dim
        self.blocks = nn.ModuleList()
        self.num_resnet_blocks: int = num_resnet_blocks
        self.batch_norm = batch_norm

        # first two hidden layers
        self.fc1 = nn.Linear(self.state_dim * self.one_hot_depth, h1_dim)
  
        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(h1_dim)

        self.fc2 = nn.Linear(h1_dim, resnet_dim)

        if self.batch_norm:
            self.bn2 = nn.BatchNorm1d(resnet_dim)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            if self.batch_norm:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_bn1 = nn.BatchNorm1d(resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                res_bn2 = nn.BatchNorm1d(resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_bn1, res_fc2, res_bn2]))
            else:
                res_fc1 = nn.Linear(resnet_dim, resnet_dim)
                res_fc2 = nn.Linear(resnet_dim, resnet_dim)
                self.blocks.append(nn.ModuleList([res_fc1, res_fc2]))

        # output
        self.fc_out = nn.Linear(resnet_dim, out_dim)

    def forward(self, states_nnet):
        x = states_nnet

        # preprocess input
        if self.one_hot_depth > 0:
            x = F.one_hot(x.long(), self.one_hot_depth)
            x = x.float()
            x = x.view(-1, self.state_dim * self.one_hot_depth)
        else:
            x = x.float()

        # first two hidden layers
        x = self.fc1(x)
        if self.batch_norm:
            x = self.bn1(x)

        x = F.relu(x)
        x = self.fc2(x)
        if self.batch_norm:
            x = self.bn2(x)

        x = F.relu(x)

        # resnet blocks
        for block_num in range(self.num_resnet_blocks):
            res_inp = x
            if self.batch_norm:
                x = self.blocks[block_num][0](x)
                x = self.blocks[block_num][1](x)
                x = F.relu(x)
                x = self.blocks[block_num][2](x)
                x = self.blocks[block_num][3](x)
            else:
                x = self.blocks[block_num][0](x)
                x = F.relu(x)
                x = self.blocks[block_num][1](x)

            x = F.relu(x + res_inp)

        # output
        x = self.fc_out(x)
        return x

# load model
def load_model():

    state_dim = 54
    nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True).to(device)
    model = "saved_models/model_state_dict.pt"

    state_dict = torch.load(model, map_location=device)
    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
                k = re.sub('^module\.', '', k)
                new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)
        
    # load model
    nnet.eval()
    return nnet

class Move(IntEnum):
    """The moves in the faceturn metric. Not to be confused with the names of the facelet positions in class Facelet."""
    U1 = 0 # U(p) face clockwise
    U3 = 1 # U(p) face counter-clockwise
    R1 = 2 # R(ight) face clockwise
    R3 = 3 # R(ight) face counter-clockwise
    F1 = 4 # F(ront) face clockwise
    F3 = 5 # F(ront) face counter-clockwise
    D1 = 6 # D(own) face clockwise
    D3 = 7 # D(own) face counter-clockwise
    L1 = 8 # L(eft) face clockwise
    L3 = 9 # L(eft) face counter-clockwise
    B1 = 10 # B(ack) face clockwise
    B3 = 11 # B(ack) face counter-clockwise

move_dict = {
    Move.U1: "U",
    Move.U3: "U'",
    Move.R1: "R",
    Move.R3: "R'",
    Move.F1: "F",
    Move.F3: "F'",
    Move.D1: "D",
    Move.D3: "D'",
    Move.L1: "L",
    Move.L3: "L'",
    Move.B1: "B",
    Move.B3: "B'",
}

single_move = [
    Move.U1, Move.U3, Move.R1, Move.R3, Move.F1, Move.F3, Move.D1, Move.D3, Move.L1, Move.L3, Move.B1, Move.B3
]

inverse_moves = {
    Move.U1 : Move.U3,
    Move.R1 : Move.R3,
    Move.F1 : Move.F3,
    Move.D1 : Move.D3,
    Move.L1 : Move.L3,
    Move.B1 : Move.B3,
    Move.U3 : Move.U1,
    Move.R3 : Move.R1,
    Move.F3 : Move.F1,
    Move.D3 : Move.D1,
    Move.L3 : Move.L1,
    Move.B3 : Move.B1
}

groups = {
    Move.U1 : "UD",
    Move.R1 : "LR",
    Move.F1 : "FB",
    Move.D1 : "UD",
    Move.L1 : "LR",
    Move.B1 : "FB",
    Move.U3 : "UD",
    Move.R3 : "LR",
    Move.F3 : "FB",
    Move.D3 : "UD",
    Move.L3 : "LR",
    Move.B3 : "FB",
}

def get_last_move(move_sequence):
    # select the last move in the sequence other than the null move. Return both the last move and the index to it
    # Otherwise, return the null move
    if len(move_sequence) == 0:
        raise ValueError("Empty move sequence")
    
    for move in (reversed(move_sequence)):
            return move

# Helper functions not provided:
def get_move_group(move):
    # Return the group number for the given move
    return groups[move]


def prevent_moves_pre(move_sequence, last_move, allowed_moves):
    temp = [move for move in move_sequence]
    subsequence = []
    last_group = get_move_group(last_move)
    
    for move in reversed(temp):
        if get_move_group(move) == last_group:
            subsequence.append(move)
        else:
            break
        
    pair_map = {}
    
    if last_group == "UD":
        pair_map = {Move.U1 : 0, Move.D1 : 0}
    elif last_group == "LR":
        pair_map = {Move.L1 : 0, Move.R1 : 0}
    else:
        pair_map = {Move.F1 : 0, Move.B1 : 0}
        
    for move in subsequence:
        if move in pair_map:
            pair_map[move] += 1
        if inverse_moves[move] in pair_map:
            pair_map[inverse_moves[move]] -= 1
            
    for i in pair_map:
        if pair_map[i] % 4 == 3:
            if i in allowed_moves:
                allowed_moves.remove(i)
        elif pair_map[i] % 4 == 1:
            if inverse_moves[i] in allowed_moves:
                allowed_moves.remove(inverse_moves[i])

    return allowed_moves


def get_allowed_moves(move_sequence):
    
    allowed_moves = list(Move)  # Start with all moves allowed
    
    if not move_sequence:
        return allowed_moves
    
    last_move = get_last_move(move_sequence)
    
    allowed_moves.remove(inverse_moves[last_move])
    
    allowed_moves = prevent_moves_pre(move_sequence, last_move, allowed_moves)
    return allowed_moves


class Cube:
    """Represent a cube on the facelet level with 54 colored facelets.
    
    Colors:

            0 0 0
            0 0 0
            0 0 0
    2 2 2   5 5 5   3 3 3   4 4 4
    2 2 2   5 5 5   3 3 3   4 4 4
    2 2 2   5 5 5   3 3 3   4 4 4
            1 1 1
            1 1 1
            1 1 1

    Order of stickers on each face:

    2   5   8
    1   4   7
    0   3   6

    Indices of state (each starting with 9*(n-1)):

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
    
    def __init__(self):
        
        # Define initial and goal state
        self.reset()
        self.goal = np.arange(0, 9 * 6) // 9
        self.state = np.arange(0, 9 * 6) // 9
        
        self.sticker_replacement = {
            Move.U1:{0: 6, 1: 3, 2: 0, 3: 7, 5: 1, 6: 8, 7: 5, 8: 2, 20: 47, 23: 50, 26: 53, 29: 38, 32: 41, 35: 44, 38: 20, 41: 23, 44: 26, 47: 29, 50: 32, 53: 35},
            Move.D1:{9: 15, 10: 12, 11: 9, 12: 16, 14: 10, 15: 17, 16: 14, 17: 11, 18: 36, 21: 39, 24: 42, 27: 45, 30: 48, 33: 51, 36: 27, 39: 30, 42: 33, 45: 18, 48: 21, 51: 24},
            Move.L1:{0: 44, 1: 43, 2: 42, 9: 45, 10: 46, 11: 47, 18: 24, 19: 21, 20: 18, 21: 25, 23: 19, 24: 26, 25: 23, 26: 20, 42: 11, 43: 10, 44: 9, 45: 0, 46: 1, 47: 2},
            Move.R1:{6: 51, 7: 52, 8: 53, 15: 38, 16: 37, 17: 36, 27: 33, 28: 30, 29: 27, 30: 34, 32: 28, 33: 35, 34: 32, 35: 29, 36: 8, 37: 7, 38: 6, 51: 15, 52: 16, 53: 17},
            Move.B1:{2: 35, 5: 34, 8: 33, 9: 20, 12: 19, 15: 18, 18: 2, 19: 5, 20: 8, 33: 9, 34: 12, 35: 15, 36: 42, 37: 39, 38: 36, 39: 43, 41: 37, 42: 44, 43: 41, 44: 38},
            Move.F1:{0: 24, 3: 25, 6: 26, 11: 27, 14: 28, 17: 29, 24: 17, 25: 14, 26: 11, 27: 6, 28: 3, 29: 0, 45: 51, 46: 48, 47: 45, 48: 52, 50: 46, 51: 53, 52: 50, 53: 47}
        }
    
    def is_solved(self):
        return np.array_equal(self.state, self.goal)
    
    def reset(self):
        self.state = np.arange(0, 9 * 6) // 9        
        
    def perform_move(self, move):
        temp_values = {}
        for original, new in self.sticker_replacement[move].items():
            temp_values[original] = self.state[new]
                
            # Update the array in place
        for new, value in temp_values.items():
            self.state[new] = value
            
    def move(self, move: Move):
        if move == Move.U1:
            self.perform_move(Move.U1)
                    
        elif move == Move.U3:
            self.move(Move.U1)
            self.move(Move.U1)
            self.move(Move.U1)
            
        elif move == Move.R1:
            self.perform_move(Move.R1)
            
        elif move == Move.R3:
            self.perform_move(Move.R1)
            self.perform_move(Move.R1)
            self.perform_move(Move.R1)

        elif move == Move.F1:
            self.perform_move(Move.F1)
            
        elif move == Move.F3:
            self.perform_move(Move.F1)
            self.perform_move(Move.F1)
            self.perform_move(Move.F1)
            
        elif move == Move.D1:
            self.perform_move(Move.D1)
            
        elif move == Move.D3:
            self.perform_move(Move.D1)
            self.perform_move(Move.D1)
            self.perform_move(Move.D1)
            
        elif move == Move.L1:
            self.perform_move(Move.L1)
            
        elif move == Move.L3:
            self.perform_move(Move.L1)
            self.perform_move(Move.L1)
            self.perform_move(Move.L1)
            
        elif move == Move.B1:
            self.perform_move(Move.B1)
            
        elif move == Move.B3:
            self.perform_move(Move.B1)
            self.perform_move(Move.B1)
            self.perform_move(Move.B1)
        else:
            raise ValueError('Invalid move: ' + str(move))

    def convert_move(self, s):
        """Convert a move string to a move."""
        s = s.split(' ')

        return_list = []
        for move in s:
            return_list.append(self.__convert_single_move(move))
        return return_list

    def __convert_single_move(self, s):
        if s == 'U':
            return Move.U1
        elif s == 'U\'':
            return Move.U3
        elif s == 'R':
            return Move.R1
        elif s == 'R\'':
            return Move.R3
        elif s == 'F':
            return Move.F1
        elif s == 'F\'':
            return Move.F3
        elif s == 'D':
            return Move.D1
        elif s == 'D\'':
            return Move.D3
        elif s == 'L':
            return Move.L1
        elif s == 'L\'':
            return Move.L3
        elif s == 'B':
            return Move.B1
        elif s == 'B\'':
            return Move.B3
        else:
            return None
    
    def move_list(self, move_list):
        """Perform a list of moves on the facelet cube."""
        for move in move_list:
            self.move(move)
            
    def copy(self):
        """Return a copy of the facelet cube."""
        new_cube = Cube()
        new_cube.state = np.copy(self.state)
        return new_cube
        
    def randomize_n(self, n):
        """Randomize the facelet cube n times."""
        scramble_move = []
        
        while len(scramble_move) < n:
            allowed_moves = get_allowed_moves(scramble_move)
            scramble_move.append(random.choice(allowed_moves))
            
        self.move_list(scramble_move)
        scramble_string = ""
        for move in scramble_move:
            scramble_string += move_dict[move] + " "
        return scramble_string, scramble_move
    
    def from_state(self, state):
        self.state = state
    
    def to_string(self):
        """Return a string representation of the facelet cube."""
        state_str = ""
        # iterate over self.state, add each element to state_str
        for i in range(54):
            state_str += str(self.state[i])
        return state_str
    
    def from_string(self, state_str):
        """Set the facelet cube to the state represented by the string state_str."""
        for i in range(54):
            self.state[i] = int(state_str[i])
    
    def __hash__(self) -> int:
        return hash(tuple(self.state))
    
    def __eq__(self, other) -> bool:
        return np.array_equal(self.state, other.state)
    
    def __str__(self):
        return self.to_string()

    def __lt__(self, other):
        return self.to_string() < other.to_string()
    
if __name__ == '__main__':
    from scramble100 import scrambles
    
    test_str = scrambles[0]
    
    cube = Cube()
    cube.move_list(cube.convert_move(test_str))
    
    test = set()
    test.add(cube)
    
    test.add(cube.copy())
    
    print(cube == cube.copy())
    
    print(len(test))
    
    print(cube.to_string())
    
    dict = {}
    dict[cube.__hash__()] = 1
    
    dict[cube.copy().__hash__()] = 2
    
    print(dict)