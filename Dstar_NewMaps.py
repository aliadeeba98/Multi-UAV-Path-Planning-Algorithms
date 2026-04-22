import os
import numpy as np
import heapq
import pandas as pd
import random

# ==============================
# A* (BASE PLANNER FOR D*)
# ==============================

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    n = grid.shape[0]
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_cost = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            neighbor = (current[0]+dx, current[1]+dy)

            if 0 <= neighbor[0] < n and 0 <= neighbor[1] < n:
                if grid[neighbor] == 1:
                    continue

                new_cost = g_cost[current] + 1

                if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                    g_cost[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

    return None


# ==============================
# D*-LIKE MULTI-UAV SIMULATION
# ==============================

def simulate_multi_uav_dstar(grid, num_agents, max_steps=200):
    free = list(zip(*np.where(grid == 0)))

    starts = random.sample(free, num_agents)
    goals = random.sample(free, num_agents)

    positions = starts.copy()

    # Initial planning
    paths = [astar(grid, starts[i], goals[i]) for i in range(num_agents)]

    if any(p is None for p in paths):
        return max_steps, 0, False

    steps = 0
    collisions = 0

    for t in range(max_steps):
        next_positions = []

        for i in range(num_agents):
            if paths[i] is None or len(paths[i]) == 0:
                next_positions.append(positions[i])
            else:
                next_positions.append(paths[i][0])

        # ==========================
        # COLLISION DETECTION
        # ==========================
        collision_flag = False

        for i in range(num_agents):
            for j in range(i+1, num_agents):

                # Same cell collision
                if next_positions[i] == next_positions[j]:
                    collisions += 1
                    collision_flag = True

                # Swap collision
                if next_positions[i] == positions[j] and next_positions[j] == positions[i]:
                    collisions += 1
                    collision_flag = True

        # ==========================
        # D* REPLANNING
        # ==========================
        if collision_flag:
            for i in range(num_agents):
                new_path = astar(grid, positions[i], goals[i])
                paths[i] = new_path if new_path else []
            continue

        # ==========================
        # EXECUTION
        # ==========================
        for i in range(num_agents):
            if len(paths[i]) > 0:
                positions[i] = paths[i].pop(0)

        steps += 1

        if all(positions[i] == goals[i] for i in range(num_agents)):
            return steps, collisions, True

    return steps, collisions, False


# ==============================
# TEST FUNCTION
# ==============================

def test_dstar(grid, num_agents, episodes=1000):
    total_steps = 0
    total_collisions = 0
    success_count = 0

    for ep in range(episodes):
        steps, collisions, success = simulate_multi_uav_dstar(grid, num_agents)

        total_steps += steps
        total_collisions += collisions
        if success:
            success_count += 1

    return (
        total_steps / episodes,
        total_collisions / episodes,
        success_count / episodes
    )


# ==============================
# EXPERIMENT RUNNER
# ==============================

def run_experiments(maps):
    results = []

    for idx, grid in enumerate(maps):
        print(f"\nRunning D* on Map {idx+1}")

        for num_agents in range(1, 5):
            print(f"Agents: {num_agents}")

            avg_steps, collision_freq, success_rate = test_dstar(grid, num_agents)

            results.append({
                "Map": idx+1,
                "Num_Agents": num_agents,
                "Avg_Steps": avg_steps,
                "Collision_Frequency": collision_freq,
                "Success_Rate": success_rate
            })

            print(f"Steps: {avg_steps:.2f}, Collisions: {collision_freq:.2f}, Success: {success_rate:.2f}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dstar_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"\nResults saved to {out_path}")
    return results


# ==============================
# MAPS
# ==============================

def create_maps():
    map1 = np.array([
        [0,0,0,0,0,0,0],
        [0,1,0,0,0,1,0],
        [0,1,0,1,0,0,0],
        [0,1,0,0,0,1,0],
        [0,0,0,1,0,1,0],
        [0,1,0,0,0,1,0],
        [0,0,0,0,0,0,0],
    ])

    map2 = np.array([
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,0,0,0,0],
        [0,1,0,1,0],
        [0,0,0,0,0],
    ])

    map3 = np.array([
        [0,0,0,0,0,0],
        [0,1,0,0,1,0],
        [0,1,0,0,1,0],
        [0,0,0,0,0,0],
        [0,1,0,0,1,0],
        [0,1,0,0,1,0],
    ])

    map4 = np.array([
        [0,0,0,0,0,0,0],
        [0,1,0,0,0,1,0],
        [0,1,0,1,0,0,0],
        [0,1,0,0,0,1,0],
        [0,0,0,1,0,1,0],
        [0,1,0,0,0,1,0],
        [0,0,0,0,0,0,0],
    ])

    map5 = np.array([
        [0,0,0,0,0,1,1,0],
        [0,0,0,1,0,1,0,0],
        [0,0,1,0,0,0,0,1],
        [0,1,0,0,0,1,0,0],
        [0,0,0,1,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,1,0,0,0,0,0,1],
        [1,0,0,0,0,1,1,0],
    ])

    map6 = np.array([
        [0,0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,1,0,0,0,0],
        [0,1,0,0,1,0,1,0,0],
        [0,1,1,1,1,1,1,1,0],
        [0,1,0,0,1,0,0,1,0],
        [0,0,0,0,1,0,0,0,0],
        [0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,0],
    ])

    return [map1, map2, map3, map4, map5, map6]


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    maps = create_maps()
    results = run_experiments(maps)