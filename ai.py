# AI for Self Driving Car

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


# ── Sum-tree (backbone of Prioritized Replay) ──────────────────────────────────
#
# A complete binary tree where each leaf stores one transition's priority^alpha.
# Every internal node holds the sum of its subtree, so:
#   - updating a priority is O(log n)  — propagate change up to root
#   - stratified sampling is O(log n) — walk down guided by cumulative sums
#
# Layout (capacity = 4):
#   tree indices:  0
#                 1   2
#                3 4 5 6   ← leaves (data stored here)
#   data indices:  0 1 2 3

class SumTree:

    def __init__(self, capacity):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data      = [None] * capacity
        self.write     = 0        # circular write pointer into leaves
        self.n_entries = 0

    @property
    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        idx = self.write + self.capacity - 1   # map data index → tree index
        self.data[self.write] = data
        self.update(idx, priority)
        self.write     = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:           # propagate delta up to root iteratively
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        """Walk the tree to find the leaf whose prefix sum covers value s."""
        idx = 0
        while True:
            left  = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]

    def __len__(self):
        return self.n_entries


# ── Prioritized Experience Replay ──────────────────────────────────────────────
#
# Transitions with high |TD-error| are sampled more often (alpha controls how
# much).  Importance-sampling weights (IS-weights) correct the resulting
# gradient bias; beta anneals from beta_start → 1 so the correction is small
# early (when estimates are noisy) and exact later.

class PrioritizedReplayMemory:

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100_000):
        self.tree         = SumTree(capacity)
        self.alpha        = alpha
        self.beta_start   = beta_start
        self.beta_frames  = beta_frames
        self.frame        = 1
        self.epsilon      = 1e-5    # floor: every transition has non-zero priority
        self.max_priority = 1.0     # new transitions start with max priority

    def push(self, event):
        # Give new transitions max priority so they are sampled at least once
        self.tree.add(self.max_priority, event)

    def sample(self, batch_size):
        batch      = []
        idxs       = []
        priorities = []

        # Stratified sampling: divide [0, total] into batch_size equal segments
        segment = self.tree.total / batch_size
        beta    = min(1.0, self.beta_start +
                      self.frame * (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)
            priorities.append(priority)

        # IS-weights: w_i = (N · P(i))^{-beta} normalised by max weight
        n       = len(self.tree)
        probs   = np.array(priorities, dtype=np.float64) / self.tree.total
        weights = (n * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        tensors = [torch.cat(x, 0) for x in zip(*batch)]
        return tensors, idxs, torch.from_numpy(weights)

    def update_priorities(self, idxs, td_errors):
        for idx, err in zip(idxs, td_errors):
            p = float((abs(err) + self.epsilon) ** self.alpha)
            self.tree.update(idx, p)
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self):
        return len(self.tree)


# ── Dueling Network ────────────────────────────────────────────────────────────
#
# After the shared trunk the network splits into two heads:
#   Value     V(s)       — how good is this state, regardless of what we do?
#   Advantage A(s, a)    — how much better is action a than the average action?
#
# Combined: Q(s,a) = V(s) + A(s,a) − mean_{a′} A(s,a′)
#
# The mean subtraction makes V and A uniquely identifiable (otherwise any
# constant can be shifted between them without changing Q).  This architecture
# helps the agent learn state values without needing to evaluate every action,
# which speeds up learning in states where the choice of action matters little.

class Network(nn.Module):

    def __init__(self, input_size, nb_action):
        super().__init__()
        # Shared feature trunk
        self.fc1  = nn.Linear(input_size, 64)
        self.fc2  = nn.Linear(64, 64)
        # Value stream: V(s) → scalar
        self.val1 = nn.Linear(64, 32)
        self.val2 = nn.Linear(32, 1)
        # Advantage stream: A(s, a) → one score per action
        self.adv1 = nn.Linear(64, 32)
        self.adv2 = nn.Linear(32, nb_action)

    def forward(self, state):
        x   = F.relu(self.fc1(state))
        x   = F.relu(self.fc2(x))
        val = self.val2(F.relu(self.val1(x)))          # (batch, 1)
        adv = self.adv2(F.relu(self.adv1(x)))          # (batch, nb_action)
        return val + adv - adv.mean(dim=1, keepdim=True)


# ── Deep Q-Learning agent ──────────────────────────────────────────────────────

class Dqn:

    def __init__(self, input_size, nb_action, gamma):
        self.gamma         = gamma
        self.reward_window = deque(maxlen=1000)
        self.model         = Network(input_size, nb_action)
        self.target_model  = Network(input_size, nb_action)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.memory        = PrioritizedReplayMemory(100_000)
        self.optimizer     = optim.Adam(self.model.parameters(), lr=5e-4)
        self.last_state    = torch.zeros(input_size).unsqueeze(0)
        self.last_action   = 0
        self.last_reward   = 0
        self.steps         = 0
        self.nb_action     = nb_action
        self.random_steps  = 0

    def select_action(self, state):
        if self.random_steps > 0:
            self.random_steps -= 1
            return random.randint(0, self.nb_action - 1)
        T = max(15.0, 100.0 - self.steps * 0.002)
        with torch.no_grad():
            q_values = self.model(state)
        probs = F.softmax(q_values * T, dim=1)
        return probs.multinomial(num_samples=1).item()

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action,
              idxs, weights):
        # Q-values for the actions that were actually taken
        outputs = self.model(batch_state).gather(
            1, batch_action.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            # Double DQN:
            #   Standard DQN uses the target network for both action selection and
            #   evaluation, which tends to overestimate Q-values.
            #   Fix: online model picks the best next action (argmax) …
            best_next = self.model(batch_next_state).argmax(1, keepdim=True)
            #   … target model evaluates that action (decouples the two roles).
            next_outputs = self.target_model(batch_next_state) \
                               .gather(1, best_next).squeeze(1)

        target    = self.gamma * next_outputs + batch_reward
        td_errors = (outputs - target).detach().abs()

        # IS-weighted loss for PER: correct the sampling bias so that high-
        # priority samples do not dominate the gradient direction.
        elementwise = F.smooth_l1_loss(outputs, target, reduction='none')
        loss = (weights * elementwise).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Feed TD errors back to the replay buffer to update priorities
        self.memory.update_priorities(idxs, td_errors.cpu().numpy())

        self.steps += 1
        if self.steps % 500 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((
            self.last_state,
            new_state,
            torch.LongTensor([int(self.last_action)]),
            torch.Tensor([self.last_reward]),
        ))
        action = self.select_action(new_state)
        if len(self.memory) > 256:
            (batch_state, batch_next_state,
             batch_action, batch_reward), idxs, weights = self.memory.sample(256)
            self.learn(batch_state, batch_next_state,
                       batch_reward, batch_action, idxs, weights)
        self.last_action = action
        self.last_state  = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def get_q_values(self):
        with torch.no_grad():
            return self.model(self.last_state).squeeze(0).tolist()

    def save(self):
        torch.save({
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'steps'     : self.steps,
        }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps = checkpoint.get('steps', 0)
            self.target_model.load_state_dict(self.model.state_dict())
            print("done!")
        else:
            print("no checkpoint found...")
