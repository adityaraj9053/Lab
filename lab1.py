from collections import deque

def get_neighbors(state):
    neighbors = []
    s = list(state)
    for i in range(7):
        if s[i] == "E":
            # move east-bound to the right
            if i+1 < 7 and s[i+1] == "_":
                new = s[:]
                new[i], new[i+1] = "_", "E"
                neighbors.append("".join(new))
            if i+2 < 7 and s[i+1] in "W" and s[i+2] == "_":
                new = s[:]
                new[i], new[i+2] = "_", "E"
                neighbors.append("".join(new))
        elif s[i] == "W":
            # move west-bound to the left
            if i-1 >= 0 and s[i-1] == "_":
                new = s[:]
                new[i], new[i-1] = "_", "W"
                neighbors.append("".join(new))
            if i-2 >= 0 and s[i-1] in "E" and s[i-2] == "_":
                new = s[:]
                new[i], new[i-2] = "_", "W"
                neighbors.append("".join(new))
    return neighbors

def bfs(start, goal):
    queue = deque([(start, [start])])
    visited = set([start])
    while queue:
        state, path = queue.popleft()
        if state == goal:
            return path
        for neighbor in get_neighbors(state):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path+[neighbor]))
    return None

start = "EEE_WWW"
goal = "WWW_EEE"
solution_bfs = bfs(start, goal)
print("BFS solution (optimal, fewest steps):")
for step in solution_bfs:
    print(step)
