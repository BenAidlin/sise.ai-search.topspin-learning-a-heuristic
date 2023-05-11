import random

from heuristics import BaseHeuristic
from priorities import f_priority
from search import search
from topspin import TopSpinState

instance_1 = [1, 7, 10, 3, 6, 9, 5, 8, 2, 4, 11]  # easy instance
instance_2 = [1, 5, 11, 2, 6, 3, 9, 4, 10, 7, 8]  # hard instance

start = TopSpinState(instance_1, 4)
heuristic = BaseHeuristic(11, 4)
path, expansions = search(start, f_priority, heuristic.get_h_value)
if path is not None:
    print(expansions)
    for vertex in path:
        print(vertex)
else:
    print("unsolvable")
