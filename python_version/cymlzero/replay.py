import numpy as np
import config

class ReplayMemory:
    def __init__(self):
        self.replay_size = config.REPLAY_SIZE
        self.idx = 0 
        self.count = 0 
        self.inputs = np.empty((config.REPLAY_SIZE, config.PLANE_SIZE, config.BOARD_SIZE, config.BOARD_SIZE), dtype=np.float32)
        self.pis = np.empty((config.REPLAY_SIZE, config.ACTION_SIZE), dtype=np.float32)
        self.values = np.empty((config.REPLAY_SIZE, 1), dtype=np.float32)

    def add(self, experience):
        state, pi, value = experience

        self.inputs[self.idx] = state
        self.pis[self.idx] = pi
        self.values[self.idx] = value
        
        self.idx = (self.idx + 1) % self.replay_size
        self.count = min(self.count + 1, self.replay_size)

    def sample(self, batch_size):
        valid_size = min(self.count, self.replay_size)
        assert valid_size != 0, f"Replay memory is empty {valid_size}"
        indices = np.random.choice(valid_size, batch_size)
        
        return (
            self.inputs[indices], 
            self.pis[indices], 
            self.values[indices] 
        )
