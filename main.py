import pandas as pd
import os
from paths import dataset_eur_usd, subset_eur_usd
from utils_dataset import get_dataset,get_day,get_range_dates
from plot import plot_trades
from environment import TradingEnv
from train import train, evaluate

import csv


train_start = "2025-04-01"
train_end   = "2025-04-30"

test_start = "2025-05-01"
test_end   = "2025-05-10"

#BETWEEN

df_train = get_range_dates(get_dataset(subset_eur_usd),train_start, train_end)
df_test  = get_range_dates(get_dataset(subset_eur_usd),test_start,test_end)

folder_model_name = f"""train_{train_start.replace("-","_")}to{train_end.replace("-","_")}"""
os.makedirs(folder_model_name, exist_ok=True)

#SINGLE DAY

#df_train = get_day(get_dataset(subset_eur_usd),train_start)
#df_test  = get_day(get_dataset(subset_eur_usd),test_start)

#folder_model_name = f"""train_{train_start.replace("-","_")}"""
#os.makedirs(folder_model_name, exist_ok=True)


# 2️⃣ Crea ambiente
env_train = TradingEnv(df_train)
env_test  = TradingEnv(df_test)


# 3️⃣ Train agente

train(env_train, folder_model_name, max_episodes=200)

# 4️⃣ Valutazione
logs = evaluate(env_test,folder_model_name, n_episodes=1)

logs[-1].to_csv(os.path.join(folder_model_name, f"""results{test_start.replace("-","_")}.csv"""),index=False,quoting=csv.QUOTE_ALL,quotechar="'")

plot_trades(logs[-1],episode=1)