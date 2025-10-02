import numpy as np
import pandas as pd
from collections import deque

class TradingEnv:

    def __init__(
                   self,
                   data,
                   sl=0.001,
                   tp=0.004,
                   spread=0.0001,
                   window=60,
                   risk_lambda=0.5,
                   trade_cost=0.01,
                   holding_reward=0.005,
                   winning_reward=0.05
                 ):
        """
        Ambiente di trading stile Gym con sequenza LSTM-ready.
        Stato = deque con ultime `window` osservazioni [features normalizzate + pnl + posizione].
        """
        self.data = data
        self.sl = sl
        self.tp = tp
        self.spread = spread
        self.window = window
        self.risk_lambda = risk_lambda
        self.trade_cost = trade_cost
        self.holding_reward = holding_reward
        self.winning_reward = winning_reward

        # Colonne feature principali
        self.feature_cols = [
                               'pct_change_BidClose',
                               'velocity_fourier_1H',
                               'velocity_fourier_1G',
                               'diff_price_fourier_1H',
                               'diff_price_fourier_1G',
                               'Hour'
                            ]

        self.n_features = len(self.feature_cols)

        # Sequenza
        self.history_deque = deque(maxlen=self.window)

        # Log dettagliato
        self.history = []

        self.reset()

    # ---------------------- Reset ----------------------
    def reset(self):
        self.current_step = self.window
        self.position = 0
        self.entry_price = 0.0
        self.pnl = 0.0
        self.total_reward = 0.0
        self.reward_adj = 0.0
        self.max_equity = 0.0
        self.num_trades = 0
        self.winning_trades = 0

        self.history_deque.clear()

        # --- Features reali dallo storico ---
        df_window = self.data.iloc[self.current_step - self.window : self.current_step][self.feature_cols].astype(np.float32)
        
        # --- Extra pnl/position iniziali (tutti zero) ---
        extra_block = np.tile(np.array([self.pnl, self.position], dtype=np.float32), (self.window, 1))

        # --- Combina features + extra ---
        sequence = np.hstack([df_window, extra_block])

        # --- Riempie la deque con la sequenza iniziale ---
        for row in sequence:
            self.history_deque.append(row)

        self.history = []
        self.index = self.data.index[self.current_step]

        return self._get_state()

    # ---------------------- Stato ----------------------
    def _get_state(self):

        """
        Aggiorna la deque con l'ultima osservazione normalizzata + pnl/posizione.
        Restituisce sequenza di shape [window, n_features + 2].
        """ 

        if self.current_step == self.window:
            return np.array(self.history_deque, dtype=np.float32)

        else:

          features = self.data.iloc[self.current_step - self.window : self.current_step][self.feature_cols].iloc[-1].values.astype(np.float32)
          extra = np.array([self.pnl, self.position], dtype=np.float32)
          step_state = np.hstack([features, extra])

          self.history_deque.append(step_state)

        return np.array(self.history_deque, dtype=np.float32)

    # ---------------------- Step ----------------------
    def step(self, action):
        self.index = self.data.index[self.current_step] if self.current_step < len(self.data) else None
        price = self.data["BidClose"].iloc[self.current_step]
        reward = 0.0
        reason = "hold"
        self.prev_position = self.position
        self.pnl = (price - self.entry_price) * self.position

        # --- Gestione azioni ---
        if action == 1:  # LONG
            if self.position == 0:
                self.position = 1
                self.entry_price = price + self.spread
                reason = "open long"
            elif self.position == -1:
                reward += self.pnl
                self.position = 0
                self.entry_price = 0.0
                self.num_trades += 1
                reason = "close short"

        elif action == 2:  # SHORT
            if self.position == 0:
                self.position = -1
                self.entry_price = price - self.spread
                reason = "open short"
            elif self.position == 1:
                reward += self.pnl
                self.position = 0
                self.entry_price = 0.0
                self.num_trades += 1
                reason = "close long"

        # --- SL / TP ---
        if self.position != 0:
            if self.pnl <= -self.sl:
                reward = self.pnl
                self.position = 0
                self.entry_price = 0.0
                self.num_trades += 1
                reason = "hit stop loss"
            elif self.pnl >= self.tp:
                reward = self.pnl
                self.position = 0
                self.entry_price = 0.0
                self.num_trades += 1
                reason = "hit take profit"

        # --- Reward shaping ---
        self.total_reward += reward
        self.max_equity = max(self.max_equity, self.total_reward)
        drawdown = self.max_equity - self.total_reward
        self.reward_adj = reward - self.risk_lambda * drawdown

        if self.position != self.prev_position:
            self.reward_adj -= self.trade_cost
        if self.position != 0 and self.position == self.prev_position:
            self.reward_adj += self.holding_reward
        if reward > 0:
            self.reward_adj += self.winning_reward
            self.winning_trades += 1

        self.current_step += 1
        done = self.current_step >= len(self.data)

        # --- Log ---
        self.history.append({
            "index": self.index,
            "step": self.current_step,
            "price": price,
            "position": self.position,
            "entry_price": self.entry_price,
            "pnl": self.pnl,
            "reward": reward,
            "reward_adj": self.reward_adj,
            "total_reward": self.total_reward,
            "reason": reason
        })

        return self._get_state(), self.reward_adj, done

    # ---------------------- Log completo ----------------------
    def get_log(self):
        return pd.DataFrame(self.history)
