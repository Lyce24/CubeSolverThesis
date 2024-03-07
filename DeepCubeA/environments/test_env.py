from cube3 import Cube3

env = Cube3()

test_move = 6


print(env.moves[test_move])
cube3state = env.generate_goal_states(1)

cube3test = env.next_state(cube3state, test_move)[0]
cube3test = env.next_state(cube3test, test_move)[0]

print(env.state_to_nnet_input(cube3test))