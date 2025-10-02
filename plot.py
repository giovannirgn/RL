import pandas as pd
import os
import matplotlib.pyplot as plt

def line_chart(df,list_cols):

    colonne_da_plottare = list_cols

    # --- Crea il grafico ---
    plt.figure(figsize=(10, 5))
    for col in colonne_da_plottare:
        plt.plot(df.index, df[col], label=col)  # line plot

    # --- Personalizzazioni ---
    plt.xlabel("Datetime")
    plt.ylabel("Valore")
    plt.title("Grafico a linee delle colonne selezionate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_trades(log, episode, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)
    df = log
    plt.figure(figsize=(15,6))
    plt.plot(df['index'], df['price'], label='Price', color='blue')
    long_mask = df['position'] == 1
    short_mask = df['position'] == -1
    plt.scatter(df['index'][long_mask], df['price'][long_mask], color='green', marker='^', label='Long', s=60)
    plt.scatter(df['index'][short_mask], df['price'][short_mask], color='red', marker='v', label='Short', s=60)
#    plt.title(f"Episode {episode} Trades & Equity")
    plt.xlabel("Time")
    plt.ylabel("Price / Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"episode_{episode}.png"))
    plt.close()

