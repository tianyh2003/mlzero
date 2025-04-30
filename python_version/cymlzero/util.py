import config

def get_weight_path(weight_id: int):
    return config.CKPT_DIR + f"/weight_{weight_id}.ckpt"

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

def print_n(n): 
    for i in range(config.BOARD_SIZE):
        for j in range(config.BOARD_SIZE):
            print(n[i * config.BOARD_SIZE + j], end=" ")
        print("")
    print("")

def print_pi(pi):
    for i in range(config.BOARD_SIZE):
        for j in range(config.BOARD_SIZE):
            _pi = round(pi[i * config.BOARD_SIZE + j], 5)
            print(f"{_pi:7.5f}", end=" ")
        print("")
    print("")