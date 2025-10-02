import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from agent import DoubleDQNAgent

def plot_trades(env, episode, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    df = env.get_log()
    plt.figure(figsize=(15,6))
    plt.plot(df['index'], df['price'], label='Price', color='blue')
    long_mask = df['position'] == 1
    short_mask = df['position'] == -1
    plt.scatter(df['index'][long_mask], df['price'][long_mask], color='green', marker='^', label='Long', s=60)
    plt.scatter(df['index'][short_mask], df['price'][short_mask], color='red', marker='v', label='Short', s=60)
    plt.plot(df['index'], df['total_reward'], color='orange', linestyle='--', label='Total Reward')
    plt.title(f"Episode {episode} Trades & Equity")
    plt.xlabel("Time")
    plt.ylabel("Price / Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"episode_{episode}.png"))
    plt.close()


def train(env, folder_model_name, max_episodes=200):
    state_shape = env.reset().shape
    n_actions = 3
    agent = DoubleDQNAgent(state_shape, n_actions)
    summary_writer = tf.summary.create_file_writer(
        os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
                                                )

    # Pre-fill replay buffer
    state = env.reset()
    for _ in range(500):
        action = np.random.randint(n_actions)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, float(done))
        state = next_state if not done else env.reset()

    rewards_history = []
    best_reward = -np.inf
    best_model_path = "best_trading_agent.h5"

    for episode in range(1, max_episodes + 1):
        state = env.reset()
        ep_reward = 0
        step_loss = []
        done = False
        num_trades = 0  # conta i trade

        while not done:
            agent.total_steps += 1
            agent.update_epsilon()
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, float(done))
            ep_reward += reward

            state = next_state

            loss = agent.train_from_replay()
            if loss is not None:
                step_loss.append(loss)
            agent.soft_update_target()

        mean_loss = np.mean(step_loss) if step_loss else 0.0
        mean_last_10 = np.mean(rewards_history[-10:]) if rewards_history else ep_reward
        rewards_history.append(ep_reward)

        print(f"Ep {episode} | Reward: {ep_reward:.4f} | Real PNL: {env.total_reward} | "
              f"Trades: {env.num_trades} | %Winning {round(env.winning_trades/env.num_trades,2)*100} |"  
              f"Mean10: {mean_last_10:.4f} | Eps: {agent.epsilon:.3f} | Loss: {mean_loss:.6f}")

        # TensorBoard logging
        with summary_writer.as_default():
            tf.summary.scalar("Reward/Episode", float(ep_reward),                                      step=episode)
            tf.summary.scalar("Reward/Mean10",  float(mean_last_10),                                   step=episode)
            tf.summary.scalar("Epsilon",        float(agent.epsilon),                                  step=episode)
            tf.summary.scalar("Loss/Mean",      float(mean_loss),                                      step=episode)
            tf.summary.scalar("Trades/Episode", float(env.num_trades),                                 step=episode)
            tf.summary.scalar("%Winning",       float(round(env.winning_trades/env.num_trades,2)*100), step=episode)
            tf.summary.scalar("TotalReward/Episode", float(env.total_reward),                          step=episode)


        # Plot trades
        #plot_trades(env, episode)

        # Salva modello migliore
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(os.path.join(folder_model_name, best_model_path))
            print(f"Nuovo best model salvato con total reward {best_reward:.4f} e {env.num_trades} trades")

    summary_writer.flush()
    summary_writer.close()
    print(f"Training completato. Miglior modello salvato in {best_model_path} con reward {best_reward:.4f}")



def evaluate(env,folder_model_name, n_episodes=5, best_model_path="best_trading_agent.h5"):

    """
    Valuta l'agente utilizzando il modello con il total reward più alto.

    """
    state_shape = env.reset().shape
    n_actions = 3

     #Crea agente

    agent = DoubleDQNAgent(state_shape, n_actions)

    # Carica modello migliore
    if os.path.exists(os.path.join(folder_model_name, best_model_path)):
        agent.load(os.path.join(folder_model_name,best_model_path))
        print(f"Caricato modello migliore da {best_model_path}")
    else:
        raise FileNotFoundError(f"Il modello migliore non è stato trovato in {best_model_path}")

    logs = []

    # Evaluation
    for ep in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, greedy=True)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state
        print(f"Episode {ep + 1} | Total Reward: {env.total_reward:.4f} | Trades: {env.num_trades} | %Winning {round(env.winning_trades/env.num_trades,2)*100}%")
        logs.append(env.get_log())

    return logs