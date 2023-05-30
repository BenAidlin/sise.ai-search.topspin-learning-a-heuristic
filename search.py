import heapq

def search(start, priority_function, heuristic_function):
    open_list = []
    closed_list = set()
    expansions = 0
    g_values = {start: 0}
    f_values = {start: heuristic_function(start)}

    heapq.heappush(open_list, (f_values[start], start))

    def create_solution(end_state):
        sol = []
        f = end_state
        while f is not None:
            sol.append(f.get_state_as_list())
            f = f.get_father_state()
        sol.reverse()
        return sol


    while open_list:
        f_current, current = heapq.heappop(open_list)

        if current.is_goal():
            return create_solution(current), expansions

        closed_list.add(current)
        expansions+=1
        for neighbor in current.get_neighbors():            
            tentative_g = g_values[current] + 1

            if neighbor in closed_list:
                continue
            # not seen before, or seen with higher g value
            if neighbor not in g_values or tentative_g < g_values[neighbor]:
                g_values[neighbor] = tentative_g
                f_values[neighbor] = priority_function(tentative_g, heuristic_function(neighbor))
                heapq.heappush(open_list, (f_values[neighbor], neighbor))                
    return (None, None)