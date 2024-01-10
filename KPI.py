#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 13:01:25 2023

@author: srishmaulik
"""

import pandas as pd
import numpy as np
from numpy import inf
        

from datetime import datetime, timedelta

# df = data1['LINKUSDT']['3m']
        
def volatility(DF,stra):
    
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df = df[stra].pct_change().resample('1d').sum()
    df = pd.DataFrame(df)
    vol = df.std() * np.sqrt(252)
    return vol.iloc[-1]
        
        
def sharpe(DF,stra):
    
    rf = 0.02
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df,stra) - rf)/volatility(df,stra)
    
    if sr == -inf:
        sr = 0
        return sr
    else:
        return sr
    
    

# daily_returns(data1['ETHUSDT']['15m']['close'])

def sharpe_diff(returns, risk_free=0):
        
    
    # returns = returns.resample('1d').sum()
    adj_returns = returns - risk_free
    return (np.nanmean(adj_returns) * np.sqrt(252)) \
        / np.nanstd(adj_returns, ddof=1)


def annual_returns_diff(returns):

    # returns = returns.resample('1d').sum()
    num_years = len(returns) / 252

    cum_ret_final = (returns + 1).prod().squeeze()

    return (cum_ret_final ** (1 / num_years) - 1)

  #  annual_returns_diff(stra[1:][np.diff(df.index.values.astype('M8[D]')).astype(int)==1].pct_change())
   # annual_returns_diff(df['returns'])


def max_drawdown_diff(cum_returns):

    max_returns = np.fmax.accumulate(cum_returns)
    res = cum_returns / max_returns - 1

   

    return abs(res.min())
"""
max_drawdown_diff(df['Strategy'])
calmar_diff(df['Strategy'])
annual_returns_diff(df['returns'])
sharpe_diff(df['returns'],risk_free=0.00)



df['Strategy'].plot()
"""
# max_drawdown_diff(df['Strategy'])


# cumret = df['Strategy']
def calmar_diff(cumret):

    max_dd = max_drawdown_diff(cumret)
    
    
    if -max_dd < 0:
        return annual_returns_diff(cumret.pct_change()) / abs(max_dd)

    return np.nan

# df = data1['ETHUSDT']['1H']
def ret(Df,com=0.04):
   
    df  = Df.copy()


    #accounting for slippage and commissions
    com = com/100
    df['returns_c'] = df['returns'].copy()
    


    df['comi'] = 0
    df['returns_c'] = np.where((df['Pos'].shift(1)==0)&(df['Pos'] == 1),df['returns_c']-com,df['returns_c'])
    df['returns_c'] = np.where((df['Pos'].shift(1)==0)&(df['Pos'] == -1),df['returns_c']-com,df['returns_c'])
    df['returns_c'] = np.where((df['Pos'].shift(1)==-1)&(df['Pos'] == 1),df['returns_c']-(com*2),df['returns_c'])
    df['returns_c'] = np.where((df['Pos'].shift(1)==1)&(df['Pos'] == -1),df['returns_c']-(com*2),df['returns_c'])


    df['returns_c'] = np.where((df['Pos'].shift(1)==1)&(df['Pos'] == 0),df['returns_c']-com,df['returns_c'])
    df['returns_c'] = np.where((df['Pos'].shift(1)==-1)&(df['Pos'] == 0),df['returns_c']-com,df['returns_c'])


        
    df['Strategy_c'] = (df['returns_c'] + 1 ).cumprod()
    df['cumsum'] = df['returns_c'].cumsum()
    df['Strategy_c'].plot()
        
    return df


# 2021-01-01 05:30:59.999000


# t = np.arange(df.index[0],df.index[-1], timedelta(days=1)).astype(datetime)

def comi_diff(Df,com=0.04):
   
    df  = Df.copy()


    #accounting for slippage and commissions
    com = com/100
    df['comi_ret'] = df['returns'].copy()
    df = df[['Pos','returns','comi_ret']]
    
    df['mark'] = abs(df['Pos'].diff())
    df['comi_ret'] = df['comi_ret'] - (com * df['mark'])

    
    df['comi_stra'] = (df['comi_ret'] + 1 ).cumprod()
    df['cumsum'] = df['comi_ret'].cumsum()
    df['comi_stra'].plot()
    return df['comi_stra'].iloc[-1]

# ret(df)
# ret_diff(df)
# calmar_diff(df['Strategy'])



# df = data1['BTCUSDT']['15m']    

# long  = (df['Pos'].diff() + 1)/2


# pos2 = pos.diff()

# df = data1['ETHUSDT']['1m']

# pos2 = df['Pos']

def get_trades_diff(df):
    
    pos  = abs(df['Pos'].diff())
    return  (pos.sum())/2
    




# get_trades_diff(df)    


def get_Trades(df,pos):

    df['trades'] = 0
    df['trades'] = np.where((df[pos]==1)&(df[pos].shift(1)==0),1,df['trades'])
    df['trades'] = np.where((df[pos]==-1)&(df[pos].shift(1)==0),1,df['trades'])
    df['trades'] = np.where((df[pos]==1)&(df[pos].shift(1)==-1),1,df['trades'])
    df['trades'] = np.where((df[pos]==-1)&(df[pos].shift(1)==1),1,df['trades'])
        
    return df['trades'].sum()


# get_Trades(df,'Pos')




def CAGR(df,stra):
     "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
     df = df[stra].pct_change().resample('1d').sum()
     df = pd.DataFrame(df)
     df = (df[stra] + 1).cumprod()
     n = len(df)/252
     df = pd.DataFrame(df)
     CAGR = (df[stra][-1])**(1/n) - 1
     return CAGR
 
# CAGR(df,'Strategy')


def calmar(DF,stra):
    "function to calculate calmar ratio"
    df = DF.copy()
    clmr = CAGR(df,stra)/max_dd(df,stra)
    if clmr == -inf:
        clmr = 0
        return clmr
    else:
        return clmr
    
    
# calmar(df,'Strategy')



def max_dd(Df,stra):
    df = Df.copy()
    "function to calculate max drawdown"
    # df["daily_ret"] = DF["close"].pct_change()
    df["cum_return"] = df[stra]
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd


# stra = df['Strategy']
# hehe = stra[1:][np.diff(df.index.values.astype('M8[D]')).astype(int)==1]

# stra[1:][np.diff(df.index.values.astype('M8[D]')).astype(int)==1].pct_change()



# max_dd(df,'Strategy')

# df = data1['BTCUSDT']['15m']
# def month(ret):
#     # df = Df.copy()
#     # t = ret.resample('1M').sum()
#     # t = np.sign(t)
#     # t = t + 1
#     # t = t/2
    
#     # t = np.sign(ret.resample('1M').sum())
#     return len(t[t==1])/len(t)
    


def ratio_l(df):
    
    
    # df['Pos'] = 0
    
    diff = abs(np.sign(df['Pos'].diff()))
    diff  = pd.DataFrame(diff[diff>0])
    
    diff['Pos'] =df['Pos']
    diff['Pos'].value_counts()
   
    
    if 1 in (diff['Pos'].value_counts()) and -1 in (diff['Pos'].value_counts()):
        
        return (diff['Pos'].value_counts().loc[-1])/\
            (diff['Pos'].value_counts().loc[1])
            
            
            

    
        