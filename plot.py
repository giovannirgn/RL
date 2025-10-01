import pandas as pd
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

