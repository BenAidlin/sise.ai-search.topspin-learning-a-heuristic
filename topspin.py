class TopSpinState:

    def __init__(self, state, k=4):
        self.state = state
        self.n = len(state)
        self.k = k
        self.father_state = None

    def is_goal(self):
        return self.state == list(range(1, self.n + 1))

    def get_state_as_list(self):
        return self.state

    def get_neighbors(self):
        neighbors = []
        # move the outer ring clockwise
        neighbors.append(TopSpinState(self.state[-1:] + self.state[:-1], self.k))
        # move the outer ring counter-clockwise
        neighbors.append(TopSpinState(self.state[1:] + self.state[:1], self.k))
        # flip the first k disks on the inner ring
        if self.k > 1:
            flipped = self.state[:self.k][::-1] + self.state[self.k:]
            neighbors.append(TopSpinState(flipped, self.k))
        for n in neighbors:
            n.father_state = self
        return neighbors
    
    def get_father_state(self):
        return self.father_state