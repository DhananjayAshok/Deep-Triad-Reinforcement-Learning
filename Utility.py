def get_state_data(state):
    return state[:9].reshape((3,3)), state[9], state[-1]