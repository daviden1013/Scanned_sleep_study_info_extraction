# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import chi2_contingency
test_n = 286



def chi2(model):
  obs = np.array([[clinical_bert, test_n-clinical_bert], [model, test_n-model]])
  g, p, dof, expctd = chi2_contingency(obs)
  print(f'p={p:.4f}, adj p={min(p*3, 1):.4f}')

""" AHI """
clinical_bert = 271
bert=272
lstm = 269
RF= 268

chi2(bert)
chi2(lstm)
chi2(RF)

""" SaO2 """

clinical_bert = 262
bert = 262
lstm = 262
RF = 256

chi2(bert)
chi2(lstm)
chi2(RF)
