import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PolicyNetwork(nn.Module if TORCH_AVAILABLE else object):
    """Durum → eylem olasılıkları üreten politika ağı."""

    def __init__(self, n_states: int, n_actions: int, hidden_size: int = 64):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch yüklü değil.")
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions),
        )

    def forward(self, x):
        return F.softmax(self.network(x), dim=-1)


class PolicyGradientAgent:
    """
    REINFORCE algoritması (Monte Carlo Policy Gradient).
    Her episode sonunda toplanan ödüllere göre politikayı günceller.
    """

    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.001,
                 discount_factor: float = 0.95, hidden_size: int = 64):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch yüklü değil.")

        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = discount_factor

        self.policy_net = PolicyNetwork(n_states, n_actions, hidden_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []
        self.losses = []

    def _state_to_tensor(self, state: int):
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return torch.FloatTensor(one_hot).unsqueeze(0)

    def choose_action(self, state: int, training: bool = True):
        """Politika ağından örnekle veya greedy seç."""
        probs = self.policy_net(self._state_to_tensor(state))
        if training:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            return int(action.item()), dist.log_prob(action)
        else:
            return int(probs.argmax().item()), None

    def _compute_returns(self, rewards: list) -> list:
        """İndirimli kümülatif ödülleri hesapla ve normalize et."""
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        # Normalize: öğrenme stabilitesi için
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def _update_policy(self, log_probs: list, returns) -> float:
        """Policy gradient loss hesapla ve ağı güncelle."""
        loss = []
        for log_prob, G in zip(log_probs, returns):
            # REINFORCE: yüksek ödüllü eylemlerin olasılığını artır
            loss.append(-log_prob * G)

        self.optimizer.zero_grad()
        total_loss = torch.stack(loss).sum()
        total_loss.backward()
        # Gradient clipping: ani sıçramaları önle
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return total_loss.item()

    def train(self, env, n_episodes: int = 3000, max_steps: int = 100):
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []
        self.losses = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            rewards = []
            log_probs = []
            steps = 0
            success = False

            for step in range(max_steps):
                action, log_prob = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                rewards.append(reward)
                log_probs.append(log_prob)
                state = next_state
                steps += 1

                if terminated:
                    success = reward > 0
                    break

            # Episode bitti, politikayı güncelle
            returns = self._compute_returns(rewards)
            loss = self._update_policy(log_probs, returns)

            self.episode_rewards.append(sum(rewards))
            self.episode_steps.append(steps)
            self.success_history.append(1 if success else 0)
            self.losses.append(loss)

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
            action, _ = self.choose_action(s, training=False)
            policy.append(action)
        return np.array(policy).reshape(size, size)
