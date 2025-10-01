import pandas as pd
from paths import dataset_eur_usd
from utils_dataset import get_dataset,get_day,get_range_dates

from environment import TradingEnv
from train import train, evaluate

import csv

# 1️⃣ Carica i dati
df_train = get_range_dates(get_dataset(dataset_eur_usd),"2025-04-01","2025-04-30")
df_test  = get_range_dates(get_dataset(dataset_eur_usd),"2025-05-02","2025-05-03")

#df_train = get_day(get_dataset(dataset_eur_usd),"2025-04-01")
#df_test  = get_day(get_dataset(dataset_eur_usd),"2025-04-02")

# 2️⃣ Crea ambiente
env_train = TradingEnv(df_train)
env_test  = TradingEnv(df_test)

# 3️⃣ Train agente
#train, best_model_path = train(env_train, max_episodes=100)

# 4️⃣ Valutazione
logs = evaluate(env_test, n_episodes=1 )

logs[-1]#.to_csv("results.csv",quoting=csv.QUOTE_ALL)