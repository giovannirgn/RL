import numpy as np
import pandas as pd

class TradingEnv:

    def __init__(
                   self,
                   data,
                   sl=0.001,
                   tp=0.004,
                   spread=0.0001,
                   window=50,
                   risk_lambda=0.5,
                   trade_cost=0.01,
                   holding_reward=0.005,
                   winning_reward =0.05
                  ):

        """
        Ambiente di trading stile Gym.
        """

        self.data           = data
        self.sl             = sl
        self.tp             = tp
        self.spread         = spread
        self.window         = window
        self.risk_lambda    = risk_lambda
        self.trade_cost     = trade_cost
        self.holding_reward = holding_reward
        self.winning_reward = winning_reward
        self.history        = []
        self.reset()

    def reset(self):

        """Reset dell'ambiente all'inizio di un episodio."""
        self.current_step   = self.window
        self.position       = 0  # 0=flat, 1=long, -1=short
        self.entry_price    = 0.0
        self.pnl            = 0.0
        self.total_reward   = 0.0
        self.reward_adj     = 0
        self.max_equity     = 0.0
        self.history        = []
        self.index          = self.data.index[self.current_step]
        self.num_trades     = 0
        self.winning_trades = 0

        return self._get_state()

    def _get_state(self):

        """Ritorna lo stato normalizzato (feature + posizione)."""
        df = self.data.iloc[self.current_step - self.window : self.current_step]
        last = df.iloc[-1]

        feature_cols = [
                          "BidClose",
                          "sma_5min",
                          "four_transf_1H_top_1",
                          "four_transf_1G_top_1",
                          "four_transf_1M_top_1"
                        ]

        state_features = last[feature_cols].values.astype(np.float32)

        df_features = df[feature_cols]

        mu = 1e-8
        state_features = (state_features - df_features.min().values) / (
                          df_features.max().values - df_features.min().values + mu
                                                                         )

        state =  np.append(state_features, self.pnl)
        state =  np.append(state, self.position ).astype(np.float32)

        return state

    def step(self, action):


        """
        Esegue un passo dell'ambiente.
        action: 0=Hold, 1=Long, 2=Short
        """
        self.index = self.data.index[self.current_step] if self.current_step < len(self.data) else None

        reward = 0.0
        price = self.data["BidClose"].iloc[self.current_step]
        reason = "hold"
        self.pnl = (price - self.entry_price) * self.position
        self.prev_position = self.position


        # ----------------------
        # Gestione azioni
        # ----------------------

        if action == 1:  # LONG
            if self.position == 0:
                self.position = 1
                self.entry_price = price + self.spread
                reason = "open long (agent)"
            if self.position == -1:
                reward += self.pnl - self.spread
                self.entry_price = 0.0
                self.position = 0
                self.num_trades += 1
                reason = "close short (agent)"

        elif action == 2:  # SHORT
            if self.position == 0:
                self.position = -1
                self.entry_price = price - self.spread
                reason = "open short (agent)"
            if self.position == 1:
                reward += self.pnl - self.spread
                self.entry_price = 0.0
                self.position = 0
                self.num_trades += 1
                reason = "close long (agent)"

        # ----------------------
        # Gestione SL/TP
        # ----------------------

        if self.position != 0:

            if self.pnl <= -self.sl:
                reward = self.pnl - self.spread
                self.entry_price = 0.0
                self.position = 0
                self.num_trades += 1
                reason = "hit stop loss"

            elif self.pnl >= self.tp:
                reward = self.pnl - self.spread
                self.entry_price = 0.0
                self.position = 0
                self.num_trades += 1
                reason = "hit take profit"

        # ----------------------
        # Reward drawdown
        # ----------------------
        self.total_reward += reward
        self.max_equity = max(self.max_equity, self.total_reward)
        drawdown = self.max_equity - self.total_reward
        self.reward_adj  = reward - self.risk_lambda * drawdown

        # ----------------------
        # Penalità trade
        # ----------------------
        if self.position != self.prev_position:
            self.reward_adj  -= self.trade_cost

        # ----------------------
        # Holding reward
        # ----------------------
        if self.position != 0 and self.position == self.prev_position:
            self.reward_adj  += self.holding_reward

        # ----------------------
        # Winning reward
        # ----------------------
        if reward > 0:
            self.reward_adj     += self.winning_reward
            self.winning_trades +=1




        # ----------------------
        # Avanza passo
        # ----------------------
        self.current_step += 1
        done = self.current_step >= len(self.data)

        # Log
        self.history.append({

                               "index"       : self.index,
                               "step"        : self.current_step,
                               "price"       : price,
                               "position"    : self.position,
                               "entry_price" : self.entry_price,
                               "pnl"         : self.pnl,
                               "reward"      : reward,
                               "reward_adj"  : self.reward_adj ,
                               "total_reward": self.total_reward,
                               "reason"      : reason

                             })

        return self._get_state(), self.reward_adj , done

    def get_log(self):

        """Ritorna il log completo dell'episodio corrente."""

        return pd.DataFrame(self.history)









