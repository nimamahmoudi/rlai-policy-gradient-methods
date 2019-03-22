def convert_s_a(state, action):
    return tuple(list(state) + [action])
