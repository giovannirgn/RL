import gym
import numpy as np
from agent import DoubleDQNAgent
import time
# Soglia di reward per abilitare render
RENDER_THRESHOLD = 475
MODEL_PATH = "cartpole_best.ckpt"
EPISODES = 10
RENDER_THRESHOLD = 475  # reward minimo per attivare il rendering

def train_cartpole(episodes=2000, model_path="cartpole_best.ckpt"):
    # Ambiente senza render durante il training
    env = gym.make("CartPole-v1")
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DoubleDQNAgent(state_shape, n_actions)
    best_reward = -np.inf

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        step_loss = []

        # Decido se fare render solo alla fine dell'episodio
        render_this_episode = False

        while not done:
            # selezione azione (epsilon-greedy gestito dall'agente)
            action = agent.select_action(state)

            # passo ambiente
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # memorizza transizione
            agent.store_transition(state, action, reward, next_state, float(done))
            state = next_state

            # addestra
            loss = agent.train_from_replay()
            if loss is not None:
                step_loss.append(loss)

            # soft update target network
            agent.soft_update_target()

            # aggiorna epsilon e total_steps ad ogni step
            agent.total_steps += 1
            agent.update_epsilon()

            # render condizionato

        mean_loss = np.mean(step_loss) if step_loss else 0.0
        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f} | "
              f"Epsilon: {agent.epsilon:.3f} | Mean Loss: {mean_loss:.6f}")

        # salva modello migliore
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(MODEL_PATH)
            print(f"💾 Nuovo best model salvato con reward {best_reward:.2f}")

    env.close()



def evaluate_cartpole():
    # Crea ambiente senza render per il loop
    env = gym.make("CartPole-v1")  # training/environment normale
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DoubleDQNAgent(state_shape, n_actions)
    agent.load(MODEL_PATH)

    for ep in range(EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Salviamo gli stati per eventuale rendering
        states_buffer = []

        while not done:
            action = agent.select_action(state, greedy=True)  # greedy evaluation
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

            states_buffer.append((state, action))

        print(f"Episode {ep+1}/{EPISODES} | Total Reward: {total_reward:.2f}")

        # Se reward supera la soglia, mostro il rendering
        if total_reward >= RENDER_THRESHOLD:
            print(f"✅ Episodio {ep+1} positivo, render attivo")
            render_env = gym.make("CartPole-v1", render_mode="human")
            state, _ = render_env.reset()
            done = False
            idx = 0

            while not done and idx < len(states_buffer):
                action = states_buffer[idx][1]  # azione già calcolata
                next_state, reward, terminated, truncated, _ = render_env.step(action)
                done = terminated or truncated
                state = next_state
                idx += 1
                time.sleep(.1)





            render_env.close()

    env.close()



if __name__ == "__main__":
    #train_cartpole()
    evaluate_cartpole()





