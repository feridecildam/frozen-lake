import numpy as np


class QLearningAgent:
    """
    Tablo tabanlı Q-Learning ajanı.
    Bellman denklemi ile Q-tablosunu günceller.
    """

    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.8,
                 discount_factor: float = 0.95, epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-tablosu: satırlar durum, sütunlar eylem
        self.q_table = np.zeros((n_states, n_actions))

        # Metrik takibi
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []

    def choose_action(self, state: int, training: bool = True) -> int:
        """Epsilon-greedy politikası ile eylem seç."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, done: bool):
        """Bellman denklemi ile Q-tablosunu güncelle."""
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])

        self.q_table[state, action] += self.lr * (target - self.q_table[state, action])

    def decay_epsilon(self):
        """Her episode sonunda epsilon'u azalt."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, env, n_episodes: int = 2000, max_steps: int = 100):
        """Ajanı eğit, her episode metriklerini kaydet."""
        self.episode_rewards = []
        self.episode_steps = []
        self.success_history = []

        for episode in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            steps = 0
            success = False

            for step in range(max_steps):
                action = self.choose_action(state, training=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                steps += 1

                if terminated:
                    success = reward > 0
                    break

            self.decay_epsilon()
            self.episode_rewards.append(total_reward)
            self.episode_steps.append(steps)
            self.success_history.append(1 if success else 0)

        return {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "success_history": self.success_history,
            "final_success_rate": np.mean(self.success_history[-100:]) * 100,
        }

    def get_policy_grid(self, size: int) -> np.ndarray:
        """Her durum için en iyi eylemi döndür (grid görselleştirme için)."""
        return np.argmax(self.q_table, axis=1).reshape(size, size)

    def get_q_table(self) -> np.ndarray:
        return self.q_table.copy()
