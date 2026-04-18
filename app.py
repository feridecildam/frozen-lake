import streamlit as st
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

from agents.q_learning import QLearningAgent
from agents.dqn import DQNAgent
from agents.policy_gradient import PolicyGradientAgent

# ─────────────────────────────────────────────
# Sayfa ayarları
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Frozen Lake — RL Karşılaştırma",
    page_icon="❄️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Sabitler
# ─────────────────────────────────────────────
ACTION_ARROWS = {0: "←", 1: "↓", 2: "→", 3: "↑"}
CELL_COLORS = {
    "S": "#4CAF50",   # Başlangıç — yeşil
    "F": "#B3E5FC",   # Buz — açık mavi
    "H": "#37474F",   # Delik — koyu gri
    "G": "#FFD700",   # Hedef — altın
}
ALGO_COLORS = {
    "Q-Learning": "#2196F3",
    "Deep Q-Learning (DQN)": "#E91E63",
    "Policy Gradient": "#FF9800",
}

# ─────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ─────────────────────────────────────────────
def get_map_desc(size: int):
    """Gymnasium'un hazır haritasını al."""
    if size == 3:
        # 3x3 için özel harita (gymnasium'da yok)
        return ["SFF", "FHF", "FFG"]
    return None  # 4x4 için gymnasium varsayılanı

def make_env(size: int, slippery: bool = False):
    """Ortamı oluştur."""
    if size == 3:
        desc = get_map_desc(3)
        return gym.make("FrozenLake-v1", desc=desc, is_slippery=slippery)
    else:
        return gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery)

def get_map_matrix(size: int) -> list:
    """Harita matrisini döndür."""
    if size == 3:
        return ["SFF", "FHF", "FFG"]
    return ["SFFF", "FHFH", "FFFH", "HFFG"]

def smooth(values: list, window: int = 50) -> np.ndarray:
    """Hareketli ortalama ile pürüzsüzleştir."""
    if len(values) < window:
        return np.array(values)
    return np.convolve(values, np.ones(window) / window, mode="valid")

def make_agent(algo: str, n_states: int, n_actions: int, params: dict):
    """Seçilen algoritmaya göre ajan oluştur."""
    if algo == "Q-Learning":
        return QLearningAgent(
            n_states=n_states, n_actions=n_actions,
            learning_rate=params["lr"],
            discount_factor=params["gamma"],
            epsilon_decay=params["epsilon_decay"],
        )
    elif algo == "Deep Q-Learning (DQN)":
        return DQNAgent(
            n_states=n_states, n_actions=n_actions,
            learning_rate=params["lr"],
            discount_factor=params["gamma"],
            epsilon_decay=params["epsilon_decay"],
            batch_size=params["batch_size"],
        )
    else:
        return PolicyGradientAgent(
            n_states=n_states, n_actions=n_actions,
            learning_rate=params["lr"],
            discount_factor=params["gamma"],
        )

# ─────────────────────────────────────────────
# Görselleştirme fonksiyonları
# ─────────────────────────────────────────────
def draw_grid(size: int, policy_grid=None, agent_pos=None, title=""):
    """Frozen Lake grid'ini çiz."""
    desc = get_map_matrix(size)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for row in range(size):
        for col in range(size):
            cell = desc[row][col]
            color = CELL_COLORS.get(cell, "#B3E5FC")

            rect = mpatches.FancyBboxPatch(
                (col + 0.05, size - row - 0.95), 0.9, 0.9,
                boxstyle="round,pad=0.05",
                facecolor=color, edgecolor="white", linewidth=2,
            )
            ax.add_patch(rect)

            # Hücre etiketi
            label = cell
            ax.text(col + 0.5, size - row - 0.65, label,
                    ha="center", va="center", fontsize=10,
                    color="white" if cell == "H" else "#263238",
                    fontweight="bold")

            # Politika oku
            if policy_grid is not None and cell not in ("H", "G"):
                action = policy_grid[row][col]
                arrow = ACTION_ARROWS.get(action, "")
                ax.text(col + 0.5, size - row - 0.35, arrow,
                        ha="center", va="center", fontsize=16,
                        color="#1565C0")

            # Ajan konumu
            if agent_pos is not None and agent_pos == row * size + col:
                ax.text(col + 0.5, size - row - 0.5, "🤖",
                        ha="center", va="center", fontsize=18)

    plt.tight_layout()
    return fig


def draw_metrics(results: dict, algo_name: str, color: str):
    """Eğitim metrik grafiklerini çiz."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    fig.suptitle(f"{algo_name} — Eğitim Metrikleri", fontsize=13, fontweight="bold")

    episodes = range(len(results["episode_rewards"]))

    # 1. Başarı oranı (hareketli ortalama)
    success_smooth = smooth(results["success_history"], window=50)
    axes[0].plot(success_smooth * 100, color=color, linewidth=1.5)
    axes[0].set_title("Başarı Oranı (%)")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("%")
    axes[0].set_ylim(0, 105)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(results["success_history"][-100:]) * 100,
                    color="red", linestyle="--", alpha=0.7,
                    label=f"Son 100 ort: {np.mean(results['success_history'][-100:])*100:.1f}%")
    axes[0].legend(fontsize=8)

    # 2. Episode başına adım sayısı
    steps_smooth = smooth(results["episode_steps"], window=50)
    axes[1].plot(steps_smooth, color=color, linewidth=1.5, alpha=0.8)
    axes[1].set_title("Episode Adım Sayısı")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Adım")
    axes[1].grid(True, alpha=0.3)

    # 3. Kayıp (DQN ve PG için) veya Q-değeri dağılımı (Q-Learning için)
    if "losses" in results and any(l > 0 for l in results["losses"]):
        losses_smooth = smooth(results["losses"], window=50)
        axes[2].plot(losses_smooth, color=color, linewidth=1.5)
        axes[2].set_title("Kayıp (Loss)")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Loss")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].bar(["Başarı", "Başarısız"],
                    [np.sum(results["success_history"]),
                     len(results["success_history"]) - np.sum(results["success_history"])],
                    color=[color, "#CFD8DC"])
        axes[2].set_title("Toplam Başarı / Başarısız")
        axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig


def draw_comparison(all_results: dict):
    """Tüm algoritmaları karşılaştır."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Algoritma Karşılaştırması", fontsize=13, fontweight="bold")

    for algo, (results, color) in all_results.items():
        success_smooth = smooth(results["success_history"], window=100)
        axes[0].plot(success_smooth * 100, label=algo, color=color, linewidth=1.5)

    axes[0].set_title("Başarı Oranı Karşılaştırması")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Başarı Oranı (%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 105)

    # Final başarı oranı çubuk grafiği
    algos = list(all_results.keys())
    final_rates = [np.mean(all_results[a][0]["success_history"][-100:]) * 100 for a in algos]
    colors = [all_results[a][1] for a in algos]
    bars = axes[1].bar(algos, final_rates, color=colors, alpha=0.85, edgecolor="white")
    axes[1].set_title("Son 100 Episode Başarı Oranı")
    axes[1].set_ylabel("%")
    axes[1].set_ylim(0, 110)
    for bar, rate in zip(bars, final_rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{rate:.1f}%", ha="center", fontsize=9, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", labelsize=8)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# Demo modu: eğitilmiş ajan adım adım oynuyor
# ─────────────────────────────────────────────
def run_demo(agent, env, size: int, max_steps: int = 30):
    """Eğitilmiş ajanı adım adım göster."""
    state, _ = env.reset()
    steps = 0
    path = [state]
    done = False

    grid_placeholder = st.empty()
    info_placeholder = st.empty()

    while not done and steps < max_steps:
        fig = draw_grid(size, agent_pos=state, title=f"Demo — Adım {steps}")
        grid_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.4)

        if hasattr(agent, "choose_action"):
            if isinstance(agent, PolicyGradientAgent):
                action, _ = agent.choose_action(state, training=False)
            else:
                action = agent.choose_action(state, training=False)
        else:
            action = np.random.randint(4)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        path.append(state)
        steps += 1

    # Son kare
    fig = draw_grid(size, agent_pos=state,
                    title="🎯 Hedefe Ulaştı!" if reward > 0 else "💀 Deliğe Düştü!")
    grid_placeholder.pyplot(fig)
    plt.close(fig)
    info_placeholder.info(
        f"Demo tamamlandı — {steps} adım | "
        f"{'✅ Başarılı' if reward > 0 else '❌ Başarısız'}"
    )


# ─────────────────────────────────────────────
# ANA UYGULAMA
# ─────────────────────────────────────────────
def main():
    # ── Başlık ──
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem'>
        <h1>❄️ Frozen Lake — Pekiştirmeli Öğrenme Karşılaştırması</h1>
        <p style='color:#888'>Q-Learning · Deep Q-Learning (DQN) · Policy Gradient</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Kenar çubuğu: ayarlar ──
    with st.sidebar:
        st.header("⚙️ Ayarlar")

        size = st.radio("Grid Boyutu", [3, 4], horizontal=True,
                        format_func=lambda x: f"{x}×{x}")
        slippery = st.toggle("Kaygan zemin (is_slippery)", value=False,
                             help="Açıksa ajan bazen rastgele kayar.")

        st.divider()
        st.subheader("Algoritma Seçimi")
        selected_algos = st.multiselect(
            "Hangi algoritmalar eğitilsin?",
            options=list(ALGO_COLORS.keys()),
            default=["Q-Learning"],
        )

        st.divider()
        st.subheader("Hiperparametreler")
        n_episodes = st.slider("Episode sayısı", 500, 5000, 2000, step=500)
        lr = st.slider("Öğrenme oranı (lr)", 0.001, 1.0, 0.8, step=0.001,
                       format="%.3f")
        gamma = st.slider("İndirim faktörü (γ)", 0.5, 0.999, 0.95, step=0.001,
                          format="%.3f")
        epsilon_decay = st.slider("Epsilon azalma", 0.990, 0.9999, 0.995,
                                  step=0.0001, format="%.4f")
        batch_size = st.select_slider("Batch boyutu (DQN)", [16, 32, 64, 128], value=64)

        params = {
            "lr": lr, "gamma": gamma,
            "epsilon_decay": epsilon_decay,
            "batch_size": batch_size,
        }

        st.divider()
        train_button = st.button("🚀 Eğitimi Başlat", use_container_width=True,
                                 type="primary",
                                 disabled=len(selected_algos) == 0)

    # ── Sekme düzeni ──
    tab_grid, tab_train, tab_compare, tab_demo = st.tabs(
        ["🗺️ Harita", "📈 Eğitim Sonuçları", "📊 Karşılaştırma", "🤖 Demo"]
    )

    # ── Harita sekmesi ──
    with tab_grid:
        st.subheader(f"Frozen Lake {size}×{size} Haritası")
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = draw_grid(size, title=f"{size}×{size} Grid")
            st.pyplot(fig)
            plt.close(fig)
        with col2:
            st.markdown("""
            **Hücre Açıklamaları:**
            - 🟢 **S** — Başlangıç noktası
            - 🔵 **F** — Donmuş yüzey (güvenli)
            - ⬛ **H** — Delik (bölüm biter, ödül = 0)
            - 🟡 **G** — Hedef (ödül = 1)

            **Eylemler:**
            - ← Sol &nbsp;&nbsp; → Sağ &nbsp;&nbsp; ↑ Yukarı &nbsp;&nbsp; ↓ Aşağı

            **Kaygan zemin açıkken** ajan istediği yönde gidemeyebilir —
            çevre stokastik hale gelir ve öğrenme zorlaşır.
            """)

    # ── Eğitim mantığı ──
    if train_button and selected_algos:
        st.session_state["trained_agents"] = {}
        st.session_state["all_results"] = {}

        env = make_env(size, slippery)
        n_states = env.observation_space.n
        n_actions = env.action_space.n

        for algo in selected_algos:
            with tab_train:
                st.subheader(f"🔄 {algo} eğitiliyor...")
                progress_bar = st.progress(0, text=f"{algo} — başlıyor...")
                status = st.empty()

            try:
                agent = make_agent(algo, n_states, n_actions, params)
            except ImportError as e:
                st.error(f"❌ {algo} için gerekli paket yüklü değil: {e}")
                continue

            # Eğitimi parçalara bölerek progress göster
            chunk = max(1, n_episodes // 20)
            all_rewards, all_steps, all_success, all_losses = [], [], [], []

            temp_env = make_env(size, slippery)

            for i in range(0, n_episodes, chunk):
                ep_count = min(chunk, n_episodes - i)
                partial = agent.train(temp_env, n_episodes=ep_count)
                all_rewards += partial["episode_rewards"]
                all_steps += partial["episode_steps"]
                all_success += partial["success_history"]
                if "losses" in partial:
                    all_losses += partial["losses"]

                pct = min(int((i + ep_count) / n_episodes * 100), 100)
                curr_rate = np.mean(all_success[-100:]) * 100 if all_success else 0
                progress_bar.progress(pct / 100,
                                      text=f"{algo} — %{pct} | Başarı: {curr_rate:.1f}%")

            temp_env.close()
            results = {
                "episode_rewards": all_rewards,
                "episode_steps": all_steps,
                "success_history": all_success,
                "final_success_rate": np.mean(all_success[-100:]) * 100,
            }
            if all_losses:
                results["losses"] = all_losses

            st.session_state["trained_agents"][algo] = (agent, size)
            st.session_state["all_results"][algo] = (results, ALGO_COLORS[algo])

            with tab_train:
                progress_bar.progress(1.0, text=f"{algo} — ✅ Tamamlandı!")
                status.success(
                    f"**{algo}** eğitimi bitti — "
                    f"Son 100 episode başarı oranı: "
                    f"**{results['final_success_rate']:.1f}%**"
                )
                fig_m = draw_metrics(results, algo, ALGO_COLORS[algo])
                st.pyplot(fig_m)
                plt.close(fig_m)

                # Politika grid'i
                try:
                    policy_grid = agent.get_policy_grid(size)
                    st.markdown("**Öğrenilen Politika:**")
                    fig_p = draw_grid(size, policy_grid=policy_grid,
                                      title=f"{algo} — Politika")
                    st.pyplot(fig_p)
                    plt.close(fig_p)
                except Exception:
                    pass

        env.close()

    # ── Karşılaştırma sekmesi ──
    with tab_compare:
        if "all_results" in st.session_state and len(st.session_state["all_results"]) > 1:
            fig_c = draw_comparison(st.session_state["all_results"])
            st.pyplot(fig_c)
            plt.close(fig_c)

            st.subheader("📋 Özet Tablo")
            rows = []
            for algo, (results, _) in st.session_state["all_results"].items():
                rows.append({
                    "Algoritma": algo,
                    "Son 100 Ep. Başarı (%)": f"{results['final_success_rate']:.1f}",
                    "Ort. Adım Sayısı": f"{np.mean(results['episode_steps']):.1f}",
                    "Toplam Başarı": f"{sum(results['success_history'])} / {len(results['success_history'])}",
                })
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        elif "all_results" in st.session_state:
            st.info("Karşılaştırma için en az 2 algoritma seçin ve eğitin.")
        else:
            st.info("Önce sol menüden algoritma seçip eğitimi başlatın.")

    # ── Demo sekmesi ──
    with tab_demo:
        if "trained_agents" in st.session_state and st.session_state["trained_agents"]:
            st.subheader("🤖 Eğitilmiş Ajanı İzle")
            demo_algo = st.selectbox(
                "Hangi ajanı izlemek istersiniz?",
                options=list(st.session_state["trained_agents"].keys()),
            )
            if st.button("▶️ Demo Başlat", type="primary"):
                agent, agent_size = st.session_state["trained_agents"][demo_algo]
                demo_env = make_env(agent_size, slippery)
                run_demo(agent, demo_env, agent_size)
                demo_env.close()
        else:
            st.info("Önce sol menüden bir algoritma eğitin, sonra burada izleyin.")


if __name__ == "__main__":
    main()


