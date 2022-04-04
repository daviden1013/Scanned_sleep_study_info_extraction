# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy.stats as st

def auc(X, Y):
    return 1/(len(X)*len(Y)) * sum([kernel(x, y) for x in X for y in Y])
def kernel(X, Y):
    return .5 if Y==X else int(Y < X)
def structural_components(X, Y):
    V10 = [1/len(Y) * sum([kernel(x, y) for y in Y]) for x in X]
    V01 = [1/len(X) * sum([kernel(x, y) for x in X]) for y in Y]
    return V10, V01
    
def get_S_entry(V_A, V_B, auc_A, auc_B):
    return 1/(len(V_A)-1) * sum([(a-auc_A)*(b-auc_B) for a,b in zip(V_A, V_B)])
def z_score(var_A, var_B, covar_AB, auc_A, auc_B):
    return (auc_A - auc_B)/((var_A + var_B - 2*covar_AB)**(.5))
def group_preds_by_label(preds, actual):
  X = [p for (p, a) in zip(preds, actual) if a]
  Y = [p for (p, a) in zip(preds, actual) if not a]
  return X, Y

def p_value(preds_A, preds_B, actual):
  
  X_A, Y_A = group_preds_by_label(preds_A, actual)
  X_B, Y_B = group_preds_by_label(preds_B, actual)
  V_A10, V_A01 = structural_components(X_A, Y_A)
  V_B10, V_B01 = structural_components(X_B, Y_B)
  auc_A = auc(X_A, Y_A)
  auc_B = auc(X_B, Y_B)
  # Compute entries of covariance matrix S (covar_AB = covar_BA)
  var_A = (get_S_entry(V_A10, V_A10, auc_A, auc_A) * 1/len(V_A10)
           + get_S_entry(V_A01, V_A01, auc_A, auc_A) * 1/len(V_A01))
  var_B = (get_S_entry(V_B10, V_B10, auc_B, auc_B) * 1/len(V_B10)
           + get_S_entry(V_B01, V_B01, auc_B, auc_B) * 1/len(V_B01))
  covar_AB = (get_S_entry(V_A10, V_B10, auc_A, auc_B) * 1/len(V_A10)
              + get_S_entry(V_A01, V_B01, auc_A, auc_B) * 1/len(V_A01))
  # Two tailed test
  z = z_score(var_A, var_B, covar_AB, auc_A, auc_B)
  p = st.norm.sf(abs(z))*2
  return p, z

""" """
PATH = r'E:\David projects\sleep study autolabel\final model'
clinical_bert = pd.read_pickle(f'{PATH}\\ClinicalBERT pred.pickle')
bert = pd.read_pickle(f'{PATH}\\BERT pred.pickle')
lstm = pd.read_pickle(f'{PATH}\\\LSTM pred.pickle')
RF = pd.read_pickle(f'{PATH}\\RF pred.pickle')

test = pd.read_pickle(f'{PATH}\\test.pickle')
label = np.array([test['label']=='AHI', test['label']=='SaO2', test['label']=='Other']).T.astype(int)

""" ClinicalBERT vs. BERT """
p, z = p_value(clinical_bert['prob_AHI'], bert['prob_AHI'], label[:,0])
print(f'p={p:.4f}, adj-p={min(p*3, 1):.4f}, z={z:.4f}')
p, z = p_value(clinical_bert['prob_SaO2'], bert['prob_SaO2'], label[:,1])
print(f'p={p:.4f}, adj-p={min(p*3, 1):.4f}, z={z:.4f}')

""" CLinicalBERT vs. LSTM+word2vec """
p, z = p_value(clinical_bert['prob_AHI'], lstm['prob_AHI'], label[:,0])
print(f'p={p:.4f}, adj-p={min(p*3, 1):.4f}, z={z:.4f}')
p, z = p_value(clinical_bert['prob_SaO2'], lstm['prob_SaO2'], label[:,1])
print(f'p={p:.4f}, adj-p={min(p*3, 1):.4f}, z={z:.4f}')

""" CLinicalBERT vs. Random Forest """
p, z = p_value(clinical_bert['prob_AHI'], RF['prob_AHI'], label[:,0])
print(f'p={p:.4f}, adj-p={min(p*3, 1):.4f}, z={z:.4f}')
p, z = p_value(clinical_bert['prob_SaO2'], RF['prob_SaO2'], label[:,1])
print(f'p={p:.4f}, adj-p={min(p*3, 1):.4f}, z={z:.4f}')

