import logging
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

def plot_feat_imp(imp):
    imp = imp.sort_values('mean', ascending=True)
    ax = imp['mean'].plot(kind='barh', color='b', alpha=0.25, xerr=imp['std'],
                          error_kw={'ecolor': 'r'}, width=1)
    ax.get_yaxis().set_visible(False)
    plt.title('Mean decrease in negative log loss.')
    plt.grid()
    plt.show()


filename = '../datasets/Bagging_Optim_importance.pickle'
optimizer = pickle.load(open(filename, 'rb'))
filename = '../datasets/Bagging_Optim.pickle'
optimizer_optim = pickle.load(open(filename, 'rb'))
# print('Importance loss per feature')
# print(optimizer.importance)

# print('Most important features table')
imp_loss_mean = optimizer.importance['mean'].mean()
# print(optimizer.importance.loc[optimizer.importance['mean'] > imp_loss_mean],)

# most_imp_features = optimizer.importance.loc[optimizer.importance['mean'] > imp_loss_mean].index
# print('Most important features list: {}'.format(most_imp_features))

plot_feat_imp(optimizer.importance)

most_imp_features = optimizer.importance.loc[optimizer.importance['mean'] > imp_loss_mean].index
print(optimizer_optim.strategies[0].Xy.loc[:,most_imp_features])

most_imp_features = most_imp_features.to_numpy()
most_imp_features = most_imp_features[most_imp_features != 'trgt'] 
print(most_imp_features)

correlation_mat = optimizer.df[most_imp_features].corr()
ax = sns.heatmap(correlation_mat, annot = True)
plt.title("Correlation matrix of selected features")

bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)

plt.show()

