from normal_solver import solver
from basic_functions import move_sticker
from tqdm import tqdm
from random import randint
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

n_data = 100000
len_solutions = [0 for _ in range(20)]

def face(mov):
    return mov // 3

def axis(mov):
    return mov // 6

def process_cube(_):
    n_move = randint(1, 20)
    cube = [i // 9 for i in range(54)]
    l_mov = -1000
    for _ in range(n_move):
        mov = randint(0, 17)
        while face(mov) == face(l_mov) or (axis(mov) == axis(l_mov) and mov < l_mov):
            mov = randint(0, 17)
        cube = move_sticker(cube, mov)
        l_mov = mov
    solution = solver(cube, 0)
    return cube, solution

def save_results(results):
    os.makedirs('learn_data/phase0', exist_ok=True)
    with open('learn_data/phase0/all_data.txt', 'a') as f:
        for cube, solution in results:
            f.write(''.join([str(i) for i in cube]))
            f.write(' ')
            f.write(str(len(solution)))
            f.write('\n')
            len_solutions[len(solution)] += 1

if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        # Map process_cube function to an iterator of n_data elements
        futures = [executor.submit(process_cube, _) for _ in range(n_data)]
        results = []
        for future in tqdm(as_completed(futures), total=n_data):
            result = future.result()
            results.append(result)
        save_results(results)

print(len_solutions)
