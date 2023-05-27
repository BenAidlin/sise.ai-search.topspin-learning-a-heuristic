import numpy as np
import random

from topspin import TopSpinState

class HeuristicTrainer:
    def __init__(self, learned_heuristic):
        self._learned_heuristic = learned_heuristic

    def train(self, max_steps=10000, epochs=100):
        training_data = []
        labels = []

        for steps in range(1, max_steps + 1):
            state = self.generate_scrambled_state(steps)
            best_neighbor = self.find_best_neighbor(state)
            label = 1 if best_neighbor.is_goal() else 1 + self._learned_heuristic.get_h_value(best_neighbor)

            training_data.append(state)
            labels.append(label)

        self._learned_heuristic.train_model(training_data, labels, epochs=epochs)

    def generate_scrambled_state(self, steps):
        state = TopSpinState(list(range(1, self._learned_heuristic._n + 1)), self._learned_heuristic._k)
        for _ in range(steps):
            state = random.choice(state.get_neighbors())
        return state

    def find_best_neighbor(self, state):
        best_neighbor = None
        best_h_value = float('inf')
        for neighbor in state.get_neighbors():
            h_value = self._learned_heuristic.get_h_value(neighbor)
            if h_value < best_h_value:
                best_h_value = h_value
                best_neighbor = neighbor
        return best_neighbor
