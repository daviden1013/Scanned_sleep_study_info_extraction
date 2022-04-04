# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join

""" function """
def clean_num(t):
  try:
    return float(t.strip('%.,'))
  except ValueError:
    if t.count('.') + t.count(',') > 1:
      return float('nan')
    if ',' in t:
      return float(t.strip('%.,').replace(',', '.'))


def segmentation(input_folder,
                 output_file,
                 PATH = r'E:\David projects\sleep study autolabel'):

  label = pd.read_excel(f'{PATH}\\data\\encryptedLabel.xlsx', sheet_name='encryptedLabel')
  label = label.loc[label['AHI_num'].notna() & label['SaO2_num'].notna()]
  label_dict = {f:(ahi, o2) for f, ahi, o2 in zip(label['file'], label['AHI_num'], label['SaO2_num'])}
  
  seq_length = 21
  csv_files = sorted([f for f in listdir(f'{PATH}\\{input_folder}') if isfile(join(f'{PATH}\\{input_folder}', f))])
  
  """
  make segments
  """
  def make_seq(l):
    seq_list = []
    for i, t in enumerate(l):
      """ if the text is number """
      if set(t) <= set('0123456789,.%') and any(c.isdigit() for c in t):
        left = max(i - int(seq_length/2), 0)
        right = min(i + int(seq_length/2) + 1, len(l))
        seq_list.append((l.index[i], ' '.join(l[left:right])))
    return seq_list
  
  def make_label(file, l):
    label_list = []
    for i, t in enumerate(l):
      """ if the text is number """
      if set(t) <= set('0123456789,.%') and any(c.isdigit() for c in t):
        """ label """
        label = 'Other'
        num = clean_num(t)
        if num == label_dict[file][0]:
          label = 'AHI'
        elif num == label_dict[file][1]:
          label = 'SaO2'
        label_list.append((l.index[i], t, num, label))
    return label_list
  
  
  seq_list = []
  for c, csv in enumerate(csv_files):
    """ print progress """
    if c % 100 == 0:
      print(f'{c} ({c/len(csv_files):.2%})')
    
    """ check in labeled list """
    file = '-'.join(csv.split('-')[0:2])
    if file not in label_dict:
      continue
    
    txt = pd.read_csv(f'{PATH}\\{input_folder}\\{csv}').fillna('')
    txt = txt.loc[txt['conf'] > -1]
    """ create horizontal/ virtical seq """
    txt['hor_center'] = np.array(txt['left']) + np.array(txt['width']/2)
    txt['hor_center_grid'] = round(txt['hor_center']/100)
    
    pos = list(txt[['left', 'top', 'width', 'height']].to_records(index=False))
    page = csv.split('-')[-1].replace('.csv', '')
    hor = txt['text']
    vir = txt.sort_values(['hor_center_grid', 'top'])['text']
    
    hor_seq = make_seq(hor)
    vir_seq = sorted(make_seq(vir), key=lambda x:x[0])
    label = make_label(file, hor)
    out = [(file, p[0], p[1], p[2], p[3], page, h[1], v[1], l[1], l[2], l[3]) for p, h, v, l in zip(pos, hor_seq, vir_seq, label)]
    
    seq_list.extend(out)
    
        
  seq = pd.DataFrame(seq_list, columns=['file', 'left', 'top', 'width', 'height', 'page', 'hor_seq', 'vir_seq', 'text', 'num', 'label'])
  
  seq.to_csv(f'{PATH}\\data\\{output_file}.csv', index=False)
  print('***Segmentation finished without error***')
