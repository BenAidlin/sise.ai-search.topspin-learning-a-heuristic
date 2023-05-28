from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np


class BaseHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k

    def get_h_value(self, state):
        state_as_list = state.get_state_as_list()
        gap = 0

        if state_as_list[0] != 1:
            gap = 1

        for i in range(len(state_as_list) - 1):
            if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                gap += 1

        return gap


class AdvanceHeuristic:
    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k

    def get_h_value(self, state):
        state_as_list = state.get_state_as_list()
        gap = 0

        if state_as_list[0] != 1:
            gap = 1
        
        last_sub_series_len = 0
        k = self._k
        n = self._n
        for i in range(len(state_as_list) - 1):
            if abs(state_as_list[i] - state_as_list[i + 1]) != 1:
                gap += 1
                if last_sub_series_len < k or last_sub_series_len % k != 0 and last_sub_series_len != n - k:
                    gap += 1
                last_sub_series_len = 0
            else:
                last_sub_series_len += 1
        return gap

    def get_h_value_temp(self, state):
        k_start_index = 0
        k_end_index = self._k-1
        state_as_list = state.get_state_as_list()
        h = 0
        for i, num in enumerate(state_as_list):
            move_left_with_k = None
            move_right_with_k = None
            move_left = None
            move_right = None

            if num < i + 1:  # need go left
                move_left = (i + 1) - num
                move_right = (len(state_as_list) - 1) - i + num

                # move left with k
                if i in range(round(k_end_index/2),k_end_index + 1):
                    new_index_after_flip = self._k - (i + 1)
                    if new_index_after_flip >= num - 1:
                        move_left_with_k = (new_index_after_flip + 1) - num + 1  # the last plus 1 is the k using 
                    else:
                        move_left_with_k = (num - 1) - new_index_after_flip + 1  # the last plus 1 is the k using 
                elif i in range(round(k_end_index/2) + 1):
                    move_left_with_k = move_left
                else:
                    move_left_with_k = (i - k_end_index) + 1 + (num - 1)
                
                # move right with k
                if i in range(round(k_end_index/2),k_end_index + 1) or i > k_end_index:
                    move_right_with_k = move_right
                else:
                    new_index_after_flip = self._k - i
                    move_right_with_k = (len(state_as_list) - 1 - new_index_after_flip) - i + num + 1

            elif num > i + 1:  # need to go right
                move_left = (len(state_as_list) - 1) + i
                move_right = num - 1

                
            
            dis = min(move_left,
                      move_right,
                      move_left_with_k,
                      move_right_with_k)
            h = max(h, dis)
        return h
    
    def get_h_value_temp2(self, state):
        # get current state as a list of integers
        state_as_list = state.get_state_as_list()
        
        # calculate number of misplaced disks
        misplaced_disks = 0
        for i in range(self._n - self._k):
            if state_as_list[i] != i + 1:
                misplaced_disks += 1
                
        # calculate number of cycles
        num_cycles = 0
        visited = set()
        for i in range(self._n):
            if i not in visited:
                visited.add(i)
                j = state_as_list.index(i + 1)
                while j != i:
                    visited.add(j)
                    j = state_as_list.index(j + 1)
                num_cycles += 1
        
        # add up and return heuristic value
        return max(misplaced_disks, num_cycles)


class LearnedHeuristic:

    def __init__(self, n=11, k=4):
        self._n = n
        self._k = k
        input_shape = (n,)

        self._model = Sequential()
        self._model.add(Dense(64, activation='relu', input_shape=input_shape))
        self._model.add(Dropout(0.25))
        self._model.add(Dense(32, activation='relu'))
        self._model.add(Dropout(0.25))
        self._model.add(Dense(16, activation='relu'))
        self._model.add(Dense(1, activation='sigmoid'))

        self._model.compile(loss='mse', optimizer='adam')

        # don't forget to load the model after you trained it

    def get_h_value(self, state):
        state = np.array(state.get_state_as_list())
        state = state.reshape(1, self._n)
        return self._model.predict(state, verbose=0)[0][0]

    def train_model(self, input_data, output_labels, epochs=100):
        input_as_list = [state.get_state_as_list() for state in input_data]
        self._model.fit(input_as_list, output_labels, epochs=epochs)

    def save_model(self):
        self._model.save_weights('models/learned_heuristic.h5')

    def load_model(self):
        self._model.load_weights('models/learned_heuristic.h5')
