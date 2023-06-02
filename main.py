import random
from heuristic_trainer import HeuristicTrainer

from heuristics import BaseHeuristic, AdvanceHeuristic, LearnedHeuristic
from priorities import f_priority, h_priority, fw_priority
from search import search
from topspin import TopSpinState

import time

training_needed = False
instances_to_run = 50
base_heuristic = BaseHeuristic(11, 4)
advanced_heuristic = AdvanceHeuristic(11, 4)
learned_heuristic = LearnedHeuristic(11, 4)
try:
    learned_heuristic.load_model() # load last learned heuristic
except:
    print('no current model to load, will start from random weights')

class SearchResult:
    def __init__(self, runtime, path_length, expansions) -> None:
        self.runtime = runtime
        self.path_length = path_length
        self.expansions = expansions

def search_and_record_performance(start: TopSpinState, heuristic: any, priority, print_states: bool) -> SearchResult:
    start_time = time.time()
    path, expansions = search(start, priority, heuristic.get_h_value)
    end_time = time.time()
    elapsed_time = end_time - start_time
    if path is not None:
        path_length = 0        
        for vertex in path:
            if(print_states):
                print(vertex)
            path_length+=1
        return SearchResult(elapsed_time, path_length, expansions)
    else:
        print("unsolvable")
        return None

def generate_scrambled_state() -> TopSpinState:    
    how_many_scrambles = random.randint(1, 10000)
    print(f'Generating scrambled state with {how_many_scrambles} scrambles')
    return HeuristicTrainer(learned_heuristic).generate_scrambled_state(how_many_scrambles)
    


def train_learned_heuristic(learned_heuristic: LearnedHeuristic, times: int, epochs: int):
    # max steps that heuristic trainer will do now defined as static member inside heuristic trainer
    heuristic_trainer = HeuristicTrainer(learned_heuristic) 
    # train model
    print("-------------training learned heuristic----------------")
    for i in range(times):
        print(f"-------------training number {i} done----------------")
        heuristic_trainer.train(epochs=epochs)
    learned_heuristic.save_model()


if training_needed:
    train_learned_heuristic(learned_heuristic, 100, 100)


priorities = [('A*', f_priority), ('GBFS', h_priority), ('WA*', fw_priority)]
heuristics = [('advanced', advanced_heuristic), ('basic', base_heuristic), ('learned', learned_heuristic),]
instances = [generate_scrambled_state() for _ in range(instances_to_run)]

print(f'running {instances_to_run} of each variation')

for heuristic in heuristics:        
    for priority in priorities:        
        priority_heuristic_results = []
        for instance in enumerate(instances):
            print(f'running {priority[0]} with {heuristic[0]} and instance number {instance[0]}')
            priority_heuristic_results.append(search_and_record_performance(instance[1], heuristic[1], priority[1], False))
        
        runtime_results = [sr.runtime for sr in priority_heuristic_results if sr is not None] 
        path_length_results = [sr.path_length for sr in priority_heuristic_results if sr is not None] 
        expansions_results = [sr.expansions for sr in priority_heuristic_results if sr is not None] 
        with open('./models/results.txt', 'w') as f:
            f.write(f'{priority[0]} with {heuristic[0]} resulted in ' + 
                f'avg runtime: {sum(runtime_results) / len(runtime_results)},' + 
                f'avg path_length: {sum(path_length_results) / len(path_length_results)},' + 
                f'avg expansions: {sum(expansions_results) / len(expansions_results)}')