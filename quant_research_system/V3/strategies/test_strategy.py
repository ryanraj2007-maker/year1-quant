import pandas as pd 
from core.trade_log import create_trade
  
def run(df):
    trades=[]
    for i in range(1, len(df)-1):
       prev = df.iloc[i-1]
       curr = df.iloc[i]
       if prev.close > prev.open:
           entry = curr.open
           atr = prev.high - prev.low
           stop = entry - 1*atr
           target = entry + 2*atr
           trade = create_trade(
               entry_time=curr.name,
               exit_time=df.index[i+1],
               direction="long",
               entry_price=entry,
               exit_price=curr.close,
               stop_price=stop,
               target_price=target,
           )
           trades.append(trade)
    return trades