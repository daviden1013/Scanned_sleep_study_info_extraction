# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

PATH = r'E:\David projects\sleep study autolabel\final model'
""" load data """
lr = f'{PATH}\\LR pred.pickle'
lasso = f'{PATH}\\LASSO pred.pickle'
ridge = f'{PATH}\\Ridge pred.pickle'
svm = f'{PATH}\\SVM pred.pickle'
knn = f'{PATH}\\kNN pred.pickle'
bayes = f'{PATH}\\NaiveBayes pred.pickle'
rf = f'{PATH}\\RF pred.pickle'
lstm = f'{PATH}\\LSTM pred.pickle'
bert = f'{PATH}\\BERT pred.pickle'
clinicalbert = f'{PATH}\\ClinicalBERT pred.pickle'

test = pd.read_pickle(f'{PATH}\\test.pickle')
label = np.array([test['label']=='AHI', test['label']=='SaO2', test['label']=='Other']).T.astype(int)

""" plot """

results = [lr, lasso, ridge, svm, knn, bayes, rf, lstm, bert, clinicalbert]
labels = ['Logistic Regression', 'Lasso', 'Ridge', 'SVM', 'kNN', 'NaiveBayes', 
             'Random Forest', 'BiLSTM','BERT', 'ClinicalBERT']
colors = ['#fafa2d', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

def plot(ax, var, ylabel=None, title=None, legend=False):
  for m, l, c in zip(results, labels, colors):
    pred = pd.read_pickle(m)
    col = 0 if var=='AHI' else 1
    fp, tr, _ = metrics.roc_curve(label[:,col], pred[f'prob_{var}'])
    ax.plot(fp, tr, label=l, color=c, linewidth=1)
  
  ax.plot([0, 1], [0, 1], color='black', linestyle='--')
  if legend:
    ax.legend()
  ax.set_title(title)
  ax.set_xlabel('False positive rate')
  ax.set_ylabel(ylabel)


fig, axs = plt.subplots(1, 2)
plot(axs[0], 'AHI', ylabel='True positive rate', title='(A) AHI')
plot(axs[1], 'SaO2', title='(B) SaO2', legend=True)

plt.savefig(r'E:\David projects\sleep study autolabel\figure\ROC.png', dpi=300, bbox_inches='tight')
plt.show()

