# -*- coding: utf-8 -*-
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def Logistic_Regression(train, test, pred_file):
  
  """ remove punctuation """
  train['hor_seq'] = [''.join([' ' if s in string.punctuation else s for s in seq]) for seq in train['hor_seq']]
  test['hor_seq'] = [''.join([' ' if s in string.punctuation else s for s in seq]) for seq in test['hor_seq']]
  
  """ make features """
  def make_features(vectorizer, data):
    feature = pd.DataFrame(vectorizer.transform(data['hor_seq']).toarray(), columns=vectorizer.get_feature_names())
    feature = pd.concat([data[['left', 'top', 'width', 'height', 'page', 'num']].fillna(-99), feature], axis=1)
    for c in feature.columns:
      feature[c] = feature[c]/(feature[c].max() - feature[c].min())
    return feature
  
  vectorizer = TfidfVectorizer(token_pattern=r'\S+', max_features=400, norm=False, stop_words=stopwords.words("english"))
  vectorizer.fit(train['hor_seq'])
  train_feature = make_features(vectorizer, train)
  test_feature = make_features(vectorizer, test)
  
  """ model """
  print('Start fitting...')
  lr = LogisticRegression(random_state=123, penalty='none', max_iter=5000)
  lr.fit(train_feature, train['label'])
  print('Finished fitting...')
  
  """ save pred """
  pred = lr.predict_proba(test_feature)
  prediction = pd.DataFrame(pred, columns=['AHI', 'Other', 'SaO2'])
  prediction['pred'] = prediction.idxmax(axis=1)
  prediction['prob_AHI'] = prediction['AHI']/prediction[['AHI', 'SaO2', 'Other']].sum(axis=1)
  prediction['prob_SaO2'] = prediction['SaO2']/prediction[['AHI', 'SaO2', 'Other']].sum(axis=1)
  prediction['label'] = list(test['label'])
  prediction['pred_AHI'] = prediction['pred'] == 'AHI'
  prediction['label_AHI'] = prediction['label'] == 'AHI'
  prediction['pred_SaO2'] = prediction['pred'] == 'SaO2'
  prediction['label_SaO2'] = prediction['label'] == 'SaO2'
  
  prediction.to_pickle(f'{pred_file}.pickle')
  print('***Logistic Regression finished without error***')
