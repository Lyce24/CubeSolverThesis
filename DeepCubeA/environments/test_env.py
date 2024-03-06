from cube3 import Cube3

env = Cube3()
print(env.goal_colors)

cube3state = env.generate_goal_states(1)

cube3test = env.next_state(cube3state, 6)[0]

print(cube3test[0].colors)

print(env.state_to_nnet_input(cube3test))