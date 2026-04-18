import numpy as np
import random
from collections import deque

# PyTorch opsiyonel — yoksa uyarı ver
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class DQNNetwork(nn.Module if TORCH_AVAILABLE else object):
    """İki gizli katmanlı sinir ağı."""

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 64):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch yüklü değil. 'pip install torch' komutunu çalıştırın.")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class DQNAgent:
    """
    Deep Q-Network ajanı.
    Experience replay ve hedef ağ (target network) kullanır.
    """

    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.001,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 batch_size: int = 64, memory_size: int = 10000,
                 target_update_freq: int = 10, hidden_size: int = 64):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch yüklü değil.")

        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Ana ağ ve hedef ağ
        self.policy_net = DQNNetwork(n_states, n_actions, hidden_size)
        self.target_net = DQNNetwork(n_states, n_actions, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Replay belleği
        self.memory = deque(maxlen=memory_size)

        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []
        self.losses = []

    def _state_to_tensor(self, state: int) -> "torch.Tensor":
        """Durum indeksini one-hot vektöre çevir."""
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)

    def choose_action(self, state: int, training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            q_values = self.policy_net(self._state_to_tensor(state))
            return int(q_values.argmax().item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self) -> float:
        """Replay belleğinden örnekle ve ağı güncelle."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Batch tensorleri oluştur
        state_batch = torch.cat([self._state_to_tensor(s) for s in states])
        next_state_batch = torch.cat([self._state_to_tensor(s) for s in next_states])
        action_batch = torch.LongTensor(actions).unsqueeze(1)
        reward_batch = torch.FloatTensor(rewards)
        done_batch = torch.BoolTensor(dones)

        # Mevcut Q değerleri
        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Hedef Q değerleri (hedef ağdan)
        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(1)[0]
            next_q[done_batch] = 0.0
            target_q = reward_batch + self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, n_episodes: int = 2000, max_steps: int = 100):
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []
        self.losses = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            success = False
            episode_loss = []

            for step in range(max_steps):
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.remember(state, action, reward, next_state, done)
                loss = self.replay()
                if loss > 0:
                    episode_loss.append(loss)

                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = reward > 0
                    break

            if episode % self.target_update_freq == 0:
                self.update_target_network()

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.success_history.append(1 if success else 0)
            self.losses.append(np.mean(episode_loss) if episode_loss else 0)

        return {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "success_history": self.success_history,
            "losses": self.losses,
            "final_success_rate": np.mean(self.success_history[-100:]) * 100,
        }

    def get_policy_grid(self, size: int) -> np.ndarray:
        policy = []
        for s in range(self.n_states):
            with torch.no_grad():
                q = self.policy_net(self._state_to_tensor(s))
                policy.append(int(q.argmax().item()))
        return np.array(policy).reshape(size, size)
