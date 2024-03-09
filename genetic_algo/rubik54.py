import random
import numpy as np
from utils.cube_utils import Facelet, Color, Corner, Edge, Move, BS, cornerFacelet, edgeFacelet, cornerColor, edgeColor, move_dict, color_dict

class Cube:
    """Represent a cube on the facelet level with 54 colored facelets."""
    def __init__(self, cp=None, co=None, ep=None, eo=None):
        self.f = []
        for i in range(9):
            self.f.append(Color.U)
        for i in range(9):
            self.f.append(Color.R)
        for i in range(9):
            self.f.append(Color.F)
        for i in range(9):
            self.f.append(Color.D)
        for i in range(9):
            self.f.append(Color.L)
        for i in range(9):
            self.f.append(Color.B)
    
        """
        :param cp: corner permutation
        :param co: corner orientation
        :param ep: edge permutation
        :param eo: edge orientation
        """
        if cp is None:
            self.cp = [Corner(i) for i in range(8)]  # You may not put this as the default two lines above!
        else:
            self.cp = cp[:]
        if co is None:
            self.co = [0]*8
        else:
            self.co = co[:]
        if ep is None:
            self.ep = [Edge(i) for i in range(12)]
        else:
            self.ep = ep[:]
        if eo is None:
            self.eo = [0] * 12
        else:
            self.eo = eo[:]

        self.is_phase1_solved = False

    def __str__(self):
        return self.to_string()
    
    def move(self, move: Move):
        if move == Move.U1:
            # U face
            self._rotate_face(Color.U)
                        
            # adjacent faces
            temp = self.f[Facelet.F1:Facelet.F4]
            self.f[Facelet.F1:Facelet.F4] = self.f[Facelet.R1:Facelet.R4]
            self.f[Facelet.R1:Facelet.R4] = self.f[Facelet.B1:Facelet.B4]
            self.f[Facelet.B1:Facelet.B4] = self.f[Facelet.L1:Facelet.L4]
            self.f[Facelet.L1:Facelet.L4] = temp
        elif move == Move.U3:
            self.move(Move.U1)
            self.move(Move.U1)
            self.move(Move.U1)
            
        elif move == Move.R1:
            # R face
            self._rotate_face(Color.R)
            
            # adjacent faces
            self.f[Facelet.U3], self.f[Facelet.U6], self.f[Facelet.U9], self.f[Facelet.F3], self.f[Facelet.F6], self.f[Facelet.F9], self.f[Facelet.D3], self.f[Facelet.D6], self.f[Facelet.D9], self.f[Facelet.B7], self.f[Facelet.B4], self.f[Facelet.B1] = self.f[Facelet.F3], self.f[Facelet.F6], self.f[Facelet.F9], self.f[Facelet.D3], self.f[Facelet.D6], self.f[Facelet.D9], self.f[Facelet.B7], self.f[Facelet.B4], self.f[Facelet.B1], self.f[Facelet.U3], self.f[Facelet.U6], self.f[Facelet.U9]
        elif move == Move.R3:
            self.move(Move.R1)
            self.move(Move.R1)
            self.move(Move.R1)
            
        elif move == Move.F1:
            # F face
            self._rotate_face(Color.F)
            
            # adjacent faces
            self.f[Facelet.U7], self.f[Facelet.U8], self.f[Facelet.U9], self.f[Facelet.R1], self.f[Facelet.R4], self.f[Facelet.R7], self.f[Facelet.D1], self.f[Facelet.D2], self.f[Facelet.D3], self.f[Facelet.L3], self.f[Facelet.L6], self.f[Facelet.L9] = self.f[Facelet.L9], self.f[Facelet.L6], self.f[Facelet.L3], self.f[Facelet.U7], self.f[Facelet.U8], self.f[Facelet.U9], self.f[Facelet.R7], self.f[Facelet.R4], self.f[Facelet.R1], self.f[Facelet.D1], self.f[Facelet.D2], self.f[Facelet.D3]
        elif move == Move.F3:
            self.move(Move.F1)
            self.move(Move.F1)
            self.move(Move.F1)
            
        # D moves
        elif move == Move.D1:
            # D face
            self._rotate_face(Color.D)
            
            # adjacent faces
            temp = self.f[Facelet.F7], self.f[Facelet.F8], self.f[Facelet.F9]
            self.f[Facelet.F7], self.f[Facelet.F8], self.f[Facelet.F9] = self.f[Facelet.L7], self.f[Facelet.L8], self.f[Facelet.L9]
            self.f[Facelet.L7], self.f[Facelet.L8], self.f[Facelet.L9] = self.f[Facelet.B7], self.f[Facelet.B8], self.f[Facelet.B9]
            self.f[Facelet.B7], self.f[Facelet.B8], self.f[Facelet.B9] = self.f[Facelet.R7], self.f[Facelet.R8], self.f[Facelet.R9]
            self.f[Facelet.R7], self.f[Facelet.R8], self.f[Facelet.R9] = temp
        elif move == Move.D3:
            self.move(Move.D1)
            self.move(Move.D1)
            self.move(Move.D1)
            
        # L moves
        elif move == Move.L1:
            # L face
            self._rotate_face(Color.L)
            
            # adjacent faces
            temp = [self.f[Facelet.U1], self.f[Facelet.U4], self.f[Facelet.U7]]
            self.f[Facelet.U1], self.f[Facelet.U4], self.f[Facelet.U7] = self.f[Facelet.B9], self.f[Facelet.B6], self.f[Facelet.B3]
            self.f[Facelet.B9], self.f[Facelet.B6], self.f[Facelet.B3] = self.f[Facelet.D1], self.f[Facelet.D4], self.f[Facelet.D7]
            self.f[Facelet.D1], self.f[Facelet.D4], self.f[Facelet.D7] = self.f[Facelet.F1], self.f[Facelet.F4], self.f[Facelet.F7]
            self.f[Facelet.F1], self.f[Facelet.F4], self.f[Facelet.F7] = temp
        elif move == Move.L3:
            self.move(Move.L1)
            self.move(Move.L1)
            self.move(Move.L1)
            
        # B moves
        elif move == Move.B1:
            # B face
            self._rotate_face(Color.B)
            
            # adjacent faces
            temp = [self.f[Facelet.U1], self.f[Facelet.U2], self.f[Facelet.U3]]
            self.f[Facelet.U1], self.f[Facelet.U2], self.f[Facelet.U3] = self.f[Facelet.R3], self.f[Facelet.R6], self.f[Facelet.R9]
            self.f[Facelet.R3], self.f[Facelet.R6], self.f[Facelet.R9] = self.f[Facelet.D9], self.f[Facelet.D8], self.f[Facelet.D7]
            self.f[Facelet.D9], self.f[Facelet.D8], self.f[Facelet.D7] = self.f[Facelet.L7], self.f[Facelet.L4], self.f[Facelet.L1]
            self.f[Facelet.L7], self.f[Facelet.L4], self.f[Facelet.L1] = temp
        elif move == Move.B3:
            self.move(Move.B1)
            self.move(Move.B1)
            self.move(Move.B1)
        
        elif move == Move.N:
            return
        
        else:
            raise ValueError('Invalid move: ' + str(move))
        
    def _rotate_face(self, face):
        """Rotate a face clockwise."""
        offset = face * 9
        new_f = self.f[offset:offset+9]  # Copy the face
        self.f[offset + 0] = new_f[6]
        self.f[offset + 2] = new_f[0]
        self.f[offset + 6] = new_f[8]
        self.f[offset + 8] = new_f[2]
        
        self.f[offset + 1] = new_f[3]
        self.f[offset + 3] = new_f[7]
        self.f[offset + 5] = new_f[1]
        self.f[offset + 7] = new_f[5]
            
    def from_string(self, s):
        """Construct a facelet cube from a string. See class Facelet(IntEnum) in enums.py for string format."""
        if len(s) < 54:
            return 'Error: Cube definition string ' + s + ' contains less than 54 facelets.'
        elif len(s) > 54:
            return 'Error: Cube definition string ' + s + ' contains more than 54 facelets.'
        cnt = [0] * 6
        for i in range(54):
            if s[i] == 'U':
                self.f[i] = Color.U
                cnt[Color.U] += 1
            elif s[i] == 'R':
                self.f[i] = Color.R
                cnt[Color.R] += 1
            elif s[i] == 'F':
                self.f[i] = Color.F
                cnt[Color.F] += 1
            elif s[i] == 'D':
                self.f[i] = Color.D
                cnt[Color.D] += 1
            elif s[i] == 'L':
                self.f[i] = Color.L
                cnt[Color.L] += 1
            elif s[i] == 'B':
                self.f[i] = Color.B
                cnt[Color.B] += 1
        if all(x == 9 for x in cnt):
            return True
        else:
            return 'Error: Cube definition string ' + s + ' does not contain exactly 9 facelets of each color.'

    def to_string(self):
        """Give a string representation of the facelet cube."""
        s = ''
        for i in range(54):
            if self.f[i] == Color.U:
                s += 'U'
            elif self.f[i] == Color.R:
                s += 'R'
            elif self.f[i] == Color.F:
                s += 'F'
            elif self.f[i] == Color.D:
                s += 'D'
            elif self.f[i] == Color.L:
                s += 'L'
            elif self.f[i] == Color.B:
                s += 'B'
        return s

    def to_2dstring(self):
        """Give a 2dstring representation of a facelet cube."""
        s = self.to_string()
        r = '   ' + s[0:3] + '\n   ' + s[3:6] + '\n   ' + s[6:9] + '\n'
        r += s[36:39] + s[18:21] + s[9:12] + s[45:48] + '\n' + s[39:42] + s[21:24] + s[12:15] + s[48:51] \
            + '\n' + s[42:45] + s[24:27] + s[15:18] + s[51:54] + '\n'
        r += '   ' + s[27:30] + '\n   ' + s[30:33] + '\n   ' + s[33:36] + '\n'
        return r

    def convert_move(self, s):
        """Convert a move string to a move."""
        s = s.split(' ')
        for move in s:
            if len(move) == 2:
                if move[1] == "2":
                    # recall the move's place
                    move_idx = s.index(move)
                    single_move = move[0]
                    # remove the original 180 degree move
                    s.remove(move)
                    # insert 2 single move in its place
                    s.insert(move_idx, single_move)
                    s.insert(move_idx, single_move)
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
            
    def randomize(self):
        """Randomize the facelet cube."""
        scramble_move = []
        for _ in range(25):
            scramble_move.append(random.choice(list(Move)))
        self.move_list(scramble_move)
        scramble_string = ""
        for move in scramble_move:
            scramble_string += move_dict[move] + " "
        return scramble_string
    
    def randomize_n(self, n):
        """Randomize the facelet cube n times."""
        scramble_move = []
        for _ in range(n):
            scramble_move.append(random.choice(list(Move)))
        self.move_list(scramble_move)
        scramble_string = ""
        for move in scramble_move:
            scramble_string += move_dict[move] + " "
        return scramble_string
    
    def is_solved(self):
        """Check if the facelet cube is solved."""
        for i in range(6):
            for j in range(9):
                if self.f[i * 9] != self.f[i * 9 + j]:
                    return False
        return True
    
    def convert_mlp_input(self):
        temp_list = []
        
        for i in range(54):
            if self.f[i] == 0:
                temp_list.append(0)
            elif self.f[i] == 1:
                temp_list.append(3)
            elif self.f[i] == 2:
                temp_list.append(2)
            elif self.f[i] == 3:
                temp_list.append(5)
            elif self.f[i] == 4:
                temp_list.append(1)
            elif self.f[i] == 5:
                temp_list.append(4)
        
        temp_list = temp_list[0:9] + temp_list[36:45] + temp_list[18:27] + temp_list[9:18]+ temp_list[45:54] + temp_list[27:36]
        color_list = [color_dict[i] for i in temp_list]
        nnet_input = np.array(color_list).reshape(1, -1)[0]
        return nnet_input
        
    def copy(self):
        return np.array(self.f)
    
    
    def convert_res_input(self):
        
        temp_list = []
        for i in range(54):
            if self.f[i] == 0:
                temp_list.append(0)
            elif self.f[i] == 1:
                temp_list.append(3)
            elif self.f[i] == 2:
                temp_list.append(5)
            elif self.f[i] == 3:
                temp_list.append(1)
            elif self.f[i] == 4:
                temp_list.append(2)
            elif self.f[i] == 5:
                temp_list.append(4)
                
        temp_list = temp_list[0:9] + temp_list[27:36] + temp_list[36:45] + temp_list[9:18]+ temp_list[45:54] + temp_list[18:27]
        
        ans_list = []
        
        for i in range(6):
            for element in self.rotate_90(temp_list[i*9:i*9+9]):
                ans_list.append(element)
        
        return np.array(ans_list, dtype=np.uint8)
        
    def rotate_90(self, vector):

        # for every 6 face (9 entries in self.f)
        test_list = [] 
        test_list.append(vector[6])
        test_list.append(vector[3])
        test_list.append(vector[0])
        test_list.append(vector[7])
        test_list.append(vector[4])
        test_list.append(vector[1])
        test_list.append(vector[8])
        test_list.append(vector[5])
        test_list.append(vector[2])
        
        return test_list
    
    def to_colorcube(self):
        pass
    
    def get_phase1_state(self):
        """Return a cubie representation of the facelet cube."""
        self.cp = [-1] * 8  # invalidate corner and edge permutation
        self.ep = [-1] * 12
        for i in Corner:
            fac = cornerFacelet[i]  # facelets of corner  at position i
            ori = 0
            for ori in range(3):
                if self.f[fac[ori]] == Color.U or self.f[fac[ori]] == Color.D:
                    break
            col1 = self.f[fac[(ori + 1) % 3]]  # colors which identify the corner at position i
            col2 = self.f[fac[(ori + 2) % 3]]
            for j in Corner:
                col = cornerColor[j]  # colors of corner j
                if col1 == col[1] and col2 == col[2]:
                    self.cp[i] = j  # we have corner j in corner position i
                    self.co[i] = ori
                    break

        for i in Edge:
            for j in Edge:
                if self.f[edgeFacelet[i][0]] == edgeColor[j][0] and \
                        self.f[edgeFacelet[i][1]] == edgeColor[j][1]:
                    self.ep[i] = j
                    self.eo[i] = 0
                    break
                if self.f[edgeFacelet[i][0]] == edgeColor[j][1] and \
                        self.f[edgeFacelet[i][1]] == edgeColor[j][0]:
                    self.ep[i] = j
                    self.eo[i] = 1
                    break
        return self.co, self.eo
    
    
    def check_phase1_solved(self):
        """Check if the cubie representation of the facelet cube is solved."""
        for i in range(8):
            if self.co[i] != 0:
                return False
        for i in range(12):
            if self.eo[i] != 0:
                return False
        return True