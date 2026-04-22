import os
import numpy as np
import random
from collections import defaultdict
import pandas as pd

# ==============================
# ENVIRONMENT SETUP
# ==============================

ACTIONS = [0, 1, 2, 3, 4]  # up, down, left, right, stay
ACTION_MAP = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (0, 0)
}

class MultiUAVEnv:
    def __init__(self, grid, num_agents):
        self.grid = grid
        self.n = grid.shape[0]
        self.num_agents = num_agents

        self.start_positions = self.get_free_positions(num_agents)
        self.goal_positions = self.get_free_positions(num_agents)

    def get_free_positions(self, k):
        free = list(zip(*np.where(self.grid == 0)))
        return random.sample(free, k)

    def reset(self):
        self.positions = self.start_positions.copy()
        return self.positions

    def is_valid(self, pos):
        x, y = pos
        return 0 <= x < self.n and 0 <= y < self.n and self.grid[x, y] == 0

    def step(self, actions):
        new_positions = []
        rewards = [0] * self.num_agents
        done = [False] * self.num_agents
        collision_count = 0

        # Move agents
        for i, action in enumerate(actions):
            dx, dy = ACTION_MAP[action]
            x, y = self.positions[i]
            new_pos = (x + dx, y + dy)

            if not self.is_valid(new_pos):
                rewards[i] -= 100
                new_pos = self.positions[i]
            else:
                rewards[i] -= 1

            new_positions.append(new_pos)

        # Check inter-agent collisions
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if new_positions[i] == new_positions[j]:
                    rewards[i] -= 50
                    rewards[j] -= 50
                    collision_count += 1

                # swap collision
                if new_positions[i] == self.positions[j] and new_positions[j] == self.positions[i]:
                    rewards[i] -= 50
                    rewards[j] -= 50
                    collision_count += 1

        # Goal check
        for i in range(self.num_agents):
            if new_positions[i] == self.goal_positions[i]:
                rewards[i] += 100
                done[i] = True

        self.positions = new_positions
        return new_positions, rewards, done, collision_count


# ==============================
# Q-LEARNING AGENT
# ==============================

class QAgent:
    def __init__(self, n_actions):
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.05

    def get_state(self, pos):
        return tuple(pos)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# ==============================
# TRAINING FUNCTION
# ==============================

def train(env, agents, episodes=5000, max_steps=200):
    for ep in range(episodes):
        states = env.reset()
        states = [agents[i].get_state(states[i]) for i in range(env.num_agents)]

        episode_steps = 0
        episode_collisions = 0

        for step in range(max_steps):
            actions = [agents[i].choose_action(states[i]) for i in range(env.num_agents)]

            next_states, rewards, done, collisions = env.step(actions)
            next_states = [agents[i].get_state(next_states[i]) for i in range(env.num_agents)]

            for i in range(env.num_agents):
                agents[i].update(states[i], actions[i], rewards[i], next_states[i])

            states = next_states
            episode_steps += 1
            episode_collisions += collisions

            if all(done):
                break

        # Print after each episode
        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1} | Steps: {episode_steps} | Collisions: {episode_collisions}")

        for agent in agents:
            agent.decay_epsilon()


# ==============================
# TESTING FUNCTION
# ==============================

def test(env, agents, episodes=1000, max_steps=200):
    total_steps = 0
    total_collisions = 0
    success_count = 0

    for agent in agents:
        agent.epsilon = 0  # greedy policy

    for ep in range(episodes):
        states = env.reset()
        steps = 0
        episode_collisions = 0

        for step in range(max_steps):
            actions = [np.argmax(agent.q_table[tuple(states[i])]) for i, agent in enumerate(agents)]

            states, _, done, collisions = env.step(actions)
            episode_collisions += collisions
            steps += 1

            if all(done):
                success_count += 1
                break

        total_steps += steps
        total_collisions += episode_collisions

    avg_steps = total_steps / episodes
    collision_freq = total_collisions / episodes
    success_rate = success_count / episodes

    return avg_steps, collision_freq, success_rate


# ==============================
# EXPERIMENT RUNNER
# ==============================

def run_experiments(maps):
    results = []

    for idx, grid in enumerate(maps):
        print(f"\nRunning on Map {idx+1}")

        for num_agents in range(1, 5):
            print(f"Agents: {num_agents}")

            env = MultiUAVEnv(grid, num_agents)
            agents = [QAgent(len(ACTIONS)) for _ in range(num_agents)]

            train(env, agents, episodes=5000)
            avg_steps, collision_freq, success_rate = test(env, agents)

            result = {
                "Map": idx + 1,
                "Num_Agents": num_agents,
                "Avg_Steps": avg_steps,
                "Collision_Frequency": collision_freq,
                "Success_Rate": success_rate
            }

            results.append(result)

            print(f"Test -> Steps: {avg_steps:.2f}, Collisions: {collision_freq:.2f}, Success: {success_rate:.2f}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "q_learning_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    print(f"\nResults saved to {out_path}")

    return results


# ==============================
# LOAD MAPS (FROM YOUR IMAGES)
# ==============================

def create_maps():
    # Manually encoded based on your images (1 = obstacle, 0 = free)

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