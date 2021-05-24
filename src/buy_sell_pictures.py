import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pickle

MODEL_NAME = 'Bagging_Optim_Feat_Imp'

filename = '../datasets/{}.pickle'.format(MODEL_NAME)
optimizer = pickle.load(open(filename, 'rb'))
strategy = optimizer.strategies[0]

df = strategy.ts.df['Close'].copy().to_frame()
df['labels'] = strategy.ts.t_events

print(df)

print(df.loc[df['labels'] == 1]['labels'])

df.Close.plot(kind='line', color='blue')
df.Close.loc[df.labels == 1].plot(ls='', marker='^', markersize=7, alpha=0.75, label='Buy', color='green')
df.Close.loc[df.labels == -1].plot(ls='', marker='v', markersize=7, alpha=0.75, label='Sell', color='red')
plt.legend()
plt.xlabel('Timestamp')
plt.ylabel('Bitcoin price USD')
plt.title('Buy and sell signals')
plt.grid()
plt.show()