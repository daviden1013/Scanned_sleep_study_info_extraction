# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

PATH = r'E:\David projects\sleep study autolabel'

""" eval with test """
def evaluate_test(prediction, test, cut=0):
  prediction.fillna(0, inplace=True)
  out = {}
  test_doc_label = pd.read_pickle(f'{PATH}\\data\\test_doc_label.pickle')
  label = np.array([test['label']=='AHI', test['label']=='SaO2', test['label']=='Other']).T.astype(int)
  def SE(var):
    i = 0 if var == 'AHI' else 1
    auroc = metrics.roc_auc_score(label[:,i], prediction[f'prob_{var}'])
    Q1 = auroc/(2-auroc)
    Q2 = (2*auroc**2)/(1+auroc)
    N1 = label[:,i].sum()
    N2 = (label[:,i] == 0).sum() 
    return ((auroc*(1-auroc) + (N1-1)*(Q1-auroc**2) + (N2-1)*(Q2-auroc**2))/(N1*N2))**0.5

  
  """ seq-level """
  def seq_eval(var):
    precision = metrics.precision_score(prediction[f'label_{var}'], prediction[f'pred_{var}'])
    recall = metrics.recall_score(prediction[f'label_{var}'], prediction[f'pred_{var}'])
    f1 = metrics.f1_score(prediction[f'label_{var}'], prediction[f'pred_{var}'])
    auroc = metrics.roc_auc_score(prediction[f'label_{var}'], prediction[f'prob_{var}'])
    se = SE(var)
    precision_list, recall_list, thresholds = metrics.precision_recall_curve(prediction[f'label_{var}'], prediction[f'pred_{var}'])
    try:
      auprc = metrics.auc(precision_list, recall_list)
    except ValueError:
      auprc = None
    return {'f1':f1, 'AUROC':auroc, 'se':se, 'AUPRC':auprc, 'precision':precision, 'recall':recall}
  
  out['seq_AHI'] = seq_eval('AHI')
  out['seq_SaO2'] = seq_eval('SaO2')
  
  """ doc-level """

  def doc_eval(var='AHI'):
    doc = pd.concat([test[['file', 'hor_seq', 'num']], prediction[['prob_AHI', 'prob_SaO2']]], axis=1)
    doc_pred = doc.sort_values(['file', f'prob_{var}'], ascending=False).groupby('file').first()
    doc_pred['label'] = np.array(test_doc_label[f'{var}_num'])
    annotated = doc_pred.loc[doc_pred[f'prob_{var}'] > cut]
    could_label = annotated.shape[0]
    correct = (annotated['num'] == annotated['label']).sum()
    p = ((doc_pred['num'] == doc_pred['label']) & (doc_pred[f'prob_{var}'] > cut)).mean()
    doc_se = ((p*(1-p))/doc_pred.shape[0])**0.5
    return {'label':could_label,
            'correct':correct,
            'doc_se': doc_se}
  
  out['doc_AHI'] = doc_eval('AHI')
  out['doc_SaO2'] = doc_eval('SaO2')
  
  return out
