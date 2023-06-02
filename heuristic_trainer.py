import numpy as np
import random

from topspin import TopSpinState

class HeuristicTrainer:
    # static member of heuristic trainer    
    max_steps = 1000

    def __init__(self, learned_heuristic):
        self._learned_heuristic = learned_heuristic

    def train(self, epochs=100):
        # initiate lists with goal and 0
        training_data = [TopSpinState(list(range(1, self._learned_heuristic._n + 1)), self._learned_heuristic._k)]
        labels = [0]

        for steps in range(1, HeuristicTrainer.max_steps + 1):
            state = self.generate_scrambled_state(steps)
            normalized_step_score = steps / HeuristicTrainer.max_steps
            label = 0 if state.is_goal() else normalized_step_score

            training_data.append(state)
            labels.append(label)
            if(steps % 10 == 0):
                print(f'step number {steps} finished')

        self._learned_heuristic.train_model(training_data, labels, epochs=epochs)

    def generate_scrambled_state(self, steps):
        state = TopSpinState(list(range(1, self._learned_heuristic._n + 1)), self._learned_heuristic._k)
        for _ in range(steps):
            state = random.choice(state.get_neighbors())
        return state    