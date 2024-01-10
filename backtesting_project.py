#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 17:51:39 2023

@author: srishmaulik
"""

#strategy for 5 min timeframe, where trade when volatility is low
import pandas as pd
import numpy as np
import finta 
from finta import*
import yfinance as yf #yahoo finance
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


start_date = datetime.now() - timedelta(days=6)  # 6 days ago

df = yf.download('AAPL', start_date, interval = '5m' )
tf = yf.download('AAPL', start_date, interval = '1m')
#-------



#-----------
df['std'] = df['Close'].rolling(10).std()
df['std'].plot()
df['Close'].plot()
df.loc["2024-01-01":"2023-01-08"]['Close'].plot()
df.loc["2024-01-01":"2024-01-08"]['Close'].plot()

df['returns'] = df['Close'].pct_change()
df['returns'].plot()
returns = df['returns'].reset_index()['returns']
plt.plot(returns)

std = df['std'].reset_index()['std']
plt.plot(std)

close = df['Close'].reset_index()['Close']
plt.plot(close)

volatility_threshold = 0.3

df['position']=np.nan
df['position'] = np.where(df['std']>volatility_threshold, -1, df['position'])
df['position'] = np.where(df['std']<volatility_threshold, 1, df['position'])
df['position'] = np.where(df['std']==volatility_threshold, 0, df['position'])

stdddd= df[df['position']==-1]
#------
tf['std'] = tf['Close'].rolling(10).std()
stnd = tf['std'].reset_index()['std']
plt.plot(stnd)

vt = 0.2
tf['position'] = np.nan
tf['position'] = np.where(tf['std']>vt, -1, tf['position']) #checking high std
tf['position'] = np.where(tf['std']<vt, 1, tf['position'])#checking lkow std
tf['position'] = np.where(tf['std']==vt, 0, tf['position'])#checking neutral crossing


tf['5minute'] = np.nan

tf['5minute'] = df['position']
# tf['5minute'] = tf['5minute'].ffill()

tf['Aposition'] = 0 
tf['Aposition'] = np.where((tf['position']==1) & (tf['5minute']==1), 1, tf['Aposition'])#where 5min and 1min timeframe have low std


tf['returns'] = tf['Close'].pct_change()

# Moving averagecrossover, checks higher tf std dev. low std enter,

tf['MA10'] = tf['Close'].rolling(10).mean()
tf['MA20'] = tf['Close'].rolling(20).mean()
tf['long'] = 0
tf['long'] = np.where((tf['MA10']>tf['MA20']) & (tf['Aposition']==1), 1, tf['long'])

tf['returns'] = tf['Close'].pct_change()*tf['long'].shift(1)
tf['Strategy'] = (tf['returns'] + 1).cumprod()  
tf['Strategy'].plot()
kk = tf.tail(500)
