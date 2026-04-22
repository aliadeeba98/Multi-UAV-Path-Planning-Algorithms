import os
import numpy as np
import random
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set QUICK_EXPERIMENT=1 for shorter training so results CSVs finish in much less time (same schema).
_QUICK = os.environ.get("QUICK_EXPERIMENT", "").lower() in ("1", "true", "yes")
TRAIN_EPISODES = 40 if _QUICK else 5000
TEST_EPISODES = 20 if _QUICK else 1000

# ==============================
# ENVIRONMENT
# ==============================

ACTIONS = [0,1,2,3,4]
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
        x,y = pos
        return 0<=x<self.n and 0<=y<self.n and self.grid[x,y]==0

    def step(self, actions):
        new_positions=[]
        rewards=[0]*self.num_agents
        done=[False]*self.num_agents
        collision_count=0

        for i,action in enumerate(actions):
            dx,dy = ACTION_MAP[action]
            x,y = self.positions[i]
            new_pos=(x+dx,y+dy)

            goal = self.goal_positions[i]

            if not self.is_valid(new_pos):
                rewards[i] -= 100
                new_pos=self.positions[i]
            else:
                rewards[i] -= 1

            # Distance-based shaping
            dist = abs(new_pos[0]-goal[0]) + abs(new_pos[1]-goal[1])
            rewards[i] -= 0.5 * dist

            new_positions.append(new_pos)

        # Inter-agent collisions
        for i in range(self.num_agents):
            for j in range(i+1,self.num_agents):
                if new_positions[i]==new_positions[j]:
                    rewards[i]-=50
                    rewards[j]-=50
                    collision_count+=1

                if new_positions[i]==self.positions[j] and new_positions[j]==self.positions[i]:
                    rewards[i]-=50
                    rewards[j]-=50
                    collision_count+=1

        for i in range(self.num_agents):
            if new_positions[i]==self.goal_positions[i]:
                rewards[i]+=200
                done[i]=True

        self.positions=new_positions
        return new_positions,rewards,done,collision_count


# ==============================
# SAC NETWORKS
# ==============================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,action_dim)
        )

    def forward(self, state):
        return self.net(state)


# ==============================
# PSO OPTIMIZER
# ==============================

class PSOOptimizer:
    def __init__(self, action_dim, num_particles=10):
        self.num_particles = num_particles
        self.action_dim = action_dim

    def optimize(self, state, actor):
        particles = np.random.rand(self.num_particles, self.action_dim)
        velocities = np.zeros_like(particles)

        personal_best = particles.copy()
        global_best = particles[0]

        def fitness(p):
            probs = actor(torch.FloatTensor(state)).detach().numpy()
            return np.dot(p, probs)

        for _ in range(5):
            for i in range(self.num_particles):
                if fitness(particles[i]) > fitness(personal_best[i]):
                    personal_best[i] = particles[i]

            global_best = max(personal_best, key=fitness)

            for i in range(self.num_particles):
                velocities[i] = 0.5*velocities[i] + \
                                1.5*np.random.rand()*(personal_best[i]-particles[i]) + \
                                1.5*np.random.rand()*(global_best-particles[i])

                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], 0, 1)

        return np.argmax(global_best)


# ==============================
# HYBRID SAC + PSO AGENT
# ==============================

class HybridAgent:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.q1 = Critic(state_dim, action_dim)
        self.q2 = Critic(state_dim, action_dim)
        self.q1_target = Critic(state_dim, action_dim)
        self.q2_target = Critic(state_dim, action_dim)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=0.001)
        self.q1_opt = optim.Adam(self.q1.parameters(), lr=0.001)
        self.q2_opt = optim.Adam(self.q2.parameters(), lr=0.001)

        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.alpha = 0.2
        self.batch_size = 64

        self.pso = PSOOptimizer(action_dim)

    def get_state(self, pos):
        return np.array(pos, dtype=np.float32)

    def choose_action(self, state):
        state_t = torch.FloatTensor(state)

        probs = self.actor(state_t).detach().numpy()
        sac_action = np.random.choice(len(probs), p=probs)

        pso_action = self.pso.optimize(state, self.actor)

        # Hybrid fusion
        if random.random() < 0.6:
            return pso_action
        else:
            return sac_action

    def store(self, s,a,r,s_next,d):
        self.memory.append((s,a,r,s_next,d))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        s,a,r,s_next,d = zip(*batch)

        s = torch.FloatTensor(s)
        s_next = torch.FloatTensor(s_next)
        a = torch.LongTensor(a)
        r = torch.FloatTensor(r)
        d = torch.FloatTensor(d)

        with torch.no_grad():
            next_probs = self.actor(s_next)
            next_log_probs = torch.log(next_probs + 1e-8)

            q1_next = self.q1_target(s_next)
            q2_next = self.q2_target(s_next)
            q_next = torch.min(q1_next, q2_next)

            v_next = (next_probs * (q_next - self.alpha * next_log_probs)).sum(dim=1)
            target_q = r + (1-d)*self.gamma*v_next

        q1_pred = self.q1(s).gather(1, a.unsqueeze(1)).squeeze()
        q2_pred = self.q2(s).gather(1, a.unsqueeze(1)).squeeze()

        loss_q1 = F.mse_loss(q1_pred, target_q)
        loss_q2 = F.mse_loss(q2_pred, target_q)

        self.q1_opt.zero_grad()
        loss_q1.backward()
        self.q1_opt.step()

        self.q2_opt.zero_grad()
        loss_q2.backward()
        self.q2_opt.step()

        probs = self.actor(s)
        log_probs = torch.log(probs + 1e-8)

        q_vals = torch.min(self.q1(s), self.q2(s))
        actor_loss = (probs * (self.alpha*log_probs - q_vals)).sum(dim=1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft updates
        for t,s in zip(self.q1_target.parameters(), self.q1.parameters()):
            t.data.copy_(0.995*t.data + 0.005*s.data)

        for t,s in zip(self.q2_target.parameters(), self.q2.parameters()):
            t.data.copy_(0.995*t.data + 0.005*s.data)


# ==============================
# TRAINING (Curriculum)
# ==============================

def train(env, agents, episodes=5000):

    for ep in range(episodes):

        if ep < 1000:
            max_steps = 100
        elif ep < 3000:
            max_steps = 150
        else:
            max_steps = 200

        states = env.reset()
        states = [agents[i].get_state(states[i]) for i in range(env.num_agents)]

        steps = 0
        collisions = 0

        for _ in range(max_steps):
            actions = [agents[i].choose_action(states[i]) for i in range(env.num_agents)]

            next_states, rewards, done, c = env.step(actions)
            next_states = [agents[i].get_state(next_states[i]) for i in range(env.num_agents)]

            for i in range(env.num_agents):
                agents[i].store(states[i], actions[i], rewards[i], next_states[i], done[i])
                agents[i].train_step()

            states = next_states
            steps += 1
            collisions += c

            if all(done):
                break

        if (ep+1) % 100 == 0:
            print(f"Episode {ep+1} | Steps:{steps} | Collisions:{collisions}")


# ==============================
# TESTING
# ==============================

def test(env, agents, episodes=1000):
    total_steps, total_collisions, success = 0,0,0

    for _ in range(episodes):
        states = env.reset()
        steps,collisions = 0,0

        for _ in range(200):
            actions=[]
            for i,a in enumerate(agents):
                s = torch.FloatTensor(a.get_state(states[i]))
                probs = a.actor(s).detach().numpy()
                actions.append(np.argmax(probs))

            states,_,done,c = env.step(actions)
            collisions += c
            steps += 1

            if all(done):
                success += 1
                break

        total_steps += steps
        total_collisions += collisions

    return total_steps/episodes, total_collisions/episodes, success/episodes


# ==============================
# RUN EXPERIMENTS
# ==============================

def run_experiments(maps):
    results=[]

    for idx,grid in enumerate(maps):
        print(f"\nMap {idx+1}")

        for num_agents in range(1,5):
            print(f"Agents: {num_agents}")

            env = MultiUAVEnv(grid, num_agents)
            agents = [HybridAgent(2,5) for _ in range(num_agents)]

            train(env, agents, episodes=TRAIN_EPISODES)
            avg_steps, col_freq, suc_rate = test(env, agents, episodes=TEST_EPISODES)

            results.append({
                "Map": idx+1,
                "Num_Agents": num_agents,
                "Avg_Steps": avg_steps,
                "Collision_Frequency": col_freq,
                "Success_Rate": suc_rate
            })

            print(f"Steps:{avg_steps:.2f}, Collisions:{col_freq:.2f}, Success:{suc_rate:.2f}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hybrid_pso_sac_results.csv")
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

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
    return [map1, map2, map3, map4, map5, map6]  # keep your existing maps


# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    maps = create_maps()
    run_experiments(maps)