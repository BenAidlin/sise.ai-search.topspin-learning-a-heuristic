import random
from heuristic_trainer import HeuristicTrainer

from heuristics import BaseHeuristic, AdvanceHeuristic, LearnedHeuristic
from priorities import f_priority
from search import search
from topspin import TopSpinState

import time
# timeit wrapper to use in code
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"took {elapsed_time:.2f} seconds to execute.")
        return result
    return wrapper

@timeit
def search_with_time(start: TopSpinState, heuristic: any, priority, print_states: bool):
    path, expansions = search(start, priority, heuristic.get_h_value)
    if path is not None:
        print(f'expansions: {expansions}')
        path_length = 0        
        for vertex in path:
            if(print_states):
                print(vertex)
            path_length+=1
        print(f'path length: {path_length}')
    else:
        print("unsolvable")

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance

easy_instance_start = TopSpinState(instance_1, 4)
hard_instance_start = TopSpinState(instance_2, 4)
base_heuristic = BaseHeuristic(11, 4)
advanced_heuristic = AdvanceHeuristic(11, 4)

# check given instances with base and advanced heuristic    
# print("-------------easy instance base heuristic----------------")
# search_with_time(easy_instance_start, base_heuristic, f_priority, False)

# print("-------------easy instance advanced heuristic----------------")
# search_with_time(easy_instance_start, advanced_heuristic, f_priority, False)

# print("-------------hard instance base heuristic----------------")
# search_with_time(hard_instance_start, base_heuristic, f_priority, False)

# print("-------------hard instance advanced heuristic----------------")
# search_with_time(hard_instance_start, advanced_heuristic, f_priority, False)

learned_heuristic = LearnedHeuristic(11, 4)
heuristic_trainer = HeuristicTrainer(learned_heuristic)

# train model
print("-------------training learned heuristic----------------")
heuristic_trainer.train(max_steps=1000, epochs=100)
learned_heuristic.save_model()

# try model
learned_heuristic.load_model()
print("-------------easy instance learned heuristic----------------")
search_with_time(easy_instance_start, learned_heuristic, f_priority, False)
print("-------------hard instance learned heuristic----------------")
search_with_time(hard_instance_start, learned_heuristic, f_priority, False)