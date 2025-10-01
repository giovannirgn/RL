import pandas as pd
from paths import dataset_eur_usd
from utils_dataset import get_dataset,get_day, get_range_dates
from environment import TradingEnv
from plot import line_chart

df  = get_dataset(dataset_eur_usd)

df_day = get_day(df,"2025-09-11")

df_dates = get_range_dates(df, "2025-05-01","2025-05-31")

env = TradingEnv(df_dates)

line_chart(df_dates,["BidClose"])

print("Reset environment")
env.reset()
print("First step action 1: Buy ")
print(env.step(1))
print(env.get_log().iloc[-1].T)
print("Second step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)
print("Third step action 1: Buy (another) ")
print(env.step(1))
print(env.get_log().iloc[-1].T)
print("4 step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)
print("5 step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)
print("6 step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)
print("7 step action 2: Sell (close long) ")
print(env.step(2))
print(env.get_log().iloc[-1].T)


print("8 step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)
print("9 step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)
print("10 step action 0: hold ")
print(env.step(0))
print(env.get_log().iloc[-1].T)

print("11 step action 2: Sell (open short) ")
print(env.step(2))
print(env.get_log().iloc[-1].T)

print("12 step action 2: Sell (another) ")
print(env.step(2))
print(env.get_log().iloc[-1].T)

print("13 step action 1: Buy (close short) ")
print(env.step(1))
print(env.get_log().iloc[-1].T)
