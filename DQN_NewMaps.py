import numpy as np
import random
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# ==============================
# ENVIRONMENT (UNCHANGED)
# ==============================

ACTIONS = [0, 1, 2, 3, 4]
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

        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if new_positions[i] == new_positions[j]:
                    rewards[i] -= 50
                    rewards[j] -= 50
                    collision_count += 1

        for i in range(self.num_agents):
            if new_positions[i] == self.goal_positions[i]:
                rewards[i] += 100
                done[i] = True

        self.positions = new_positions
        return new_positions, rewards, done, collision_count


# ==============================
# DQN NETWORK
# ==============================

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ==============================
# DQN AGENT
# ==============================

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 64

    def get_state(self, pos):
        return np.array(pos, dtype=np.float32)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)

        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        target = q_values.clone()

        for i in range(self.batch_size):
            target[i][actions[i]] = rewards[i] + (1 - dones[i]) * self.gamma * torch.max(next_q_values[i])

        loss = nn.MSELoss()(q_values, target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


# ==============================
# TRAINING
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
                agents[i].store(states[i], actions[i], rewards[i], next_states[i], done[i])
                agents[i].train_step()

            states = next_states
            episode_steps += 1
            episode_collisions += collisions

            if all(done):
                break

        if (ep + 1) % 100 == 0:
            print(f"Episode {ep+1} | Steps: {episode_steps} | Collisions: {episode_collisions}")

        for agent in agents:
            agent.decay_epsilon()
            agent.update_target()


# ==============================
# TESTING (UNCHANGED LOGIC)
# ==============================

def test(env, agents, episodes=1000, max_steps=200):
    total_steps, total_collisions, success_count = 0, 0, 0

    for agent in agents:
        agent.epsilon = 0

    for _ in range(episodes):
        states = env.reset()
        steps, episode_collisions = 0, 0

        for _ in range(max_steps):
            actions = [agent.choose_action(agent.get_state(states[i])) for i, agent in enumerate(agents)]
            states, _, done, collisions = env.step(actions)

            steps += 1
            episode_collisions += collisions

            if all(done):
                success_count += 1
                break

        total_steps += steps
        total_collisions += episode_collisions

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
        print(f"\nRunning on Map {idx+1}")

        for num_agents in range(1, 5):
            print(f"Agents: {num_agents}")

            env = MultiUAVEnv(grid, num_agents)
            agents = [DQNAgent(state_dim=2, action_dim=5) for _ in range(num_agents)]

            train(env, agents)
            avg_steps, collision_freq, success_rate = test(env, agents)

            results.append({
                "Map": idx+1,
                "Num_Agents": num_agents,
                "Avg_Steps": avg_steps,
                "Collision_Frequency": collision_freq,
                "Success_Rate": success_rate
            })

            print(f"Test -> Steps: {avg_steps:.2f}, Collisions: {collision_freq:.2f}, Success: {success_rate:.2f}")

    df = pd.DataFrame(results)
    df.to_csv("dqn_results.csv", index=False)
    print("\nResults saved to dqn_results.csv")

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