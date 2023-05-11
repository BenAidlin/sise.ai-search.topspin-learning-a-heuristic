import heapq

def search(start, priority_function, heuristic_function):
    open_list = []
    closed_list = set()

    g_values = {start: 0}
    f_values = {start: heuristic_function.get_h_value(start)}

    heapq.heappush(open_list, (f_values[start], start))

    while open_list:
        f_current, current = heapq.heappop(open_list)

        if current.is_goal():
            return current

        closed_list.add(current)

        for neighbor in current.get_neighbors():
            tentative_g = g_values[current] + 1

            if neighbor in closed_list:
                continue
            # not seen before, or seen with higher g value
            if neighbor not in g_values or tentative_g < g_values[neighbor]:
                g_values[neighbor] = tentative_g
                f_values[neighbor] = priority_function(tentative_g, heuristic_function.get_h_value(neighbor))
                heapq.heappush(open_list, (f_values[neighbor], neighbor))

    return None