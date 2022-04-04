# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def train_test_split(input_file, output_folder, PATH = r'E:\David projects\sleep study autolabel'):
  
  """ load data """
  data = pd.read_csv(f'{PATH}\\data\\{input_file}.csv')

  """ split train, validate, test """
  def split(data, test_ratio=0.3):
    np.random.seed(123)
    files = data['file'].unique()
    test_id = np.random.choice(sorted(files), int(len(files) * test_ratio), replace=False)
    train_id = np.array([f for f in files if f not in test_id])
    
    return (data.loc[data['file'].isin(train_id)].reset_index(drop=True),
            data.loc[data['file'].isin(test_id)].reset_index(drop=True))
  
  train, test = split(data)

  train.to_pickle(f'{PATH}\\{output_folder}\\train.pickle')
  test.to_pickle(f'{PATH}\\{output_folder}\\test.pickle')
  print('***Train test split finished***')