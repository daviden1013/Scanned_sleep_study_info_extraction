# -*- coding: utf-8 -*-
"""
Use seq lenght of 21, 
split train: validate: test =6:1:3
have fixed input: left, right, top, down, page, num
     seq input: BERT tokens
BERT has 20% dropout
initial training freezes BERT, select best model (AHI F1) for finetune.
finetune freezes other layers and only train BERT. 

"""
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from transformers import BertTokenizer, TFBertModel
from tensorflow.keras.utils import plot_model
from sklearn import metrics
from sklearn.utils import class_weight
import keras.backend as K

max_length = 32
def ClinicalBERT_small(trainset, test, subset_n, model_dir, pred_file, ClinicalBERT_dir):
  """ split train, validate """
  def split(trainset, validate_ratio=0.16):
    np.random.seed(123)
    files = trainset['file'].unique()
    validate_id = np.random.choice(sorted(files), int(len(files) * validate_ratio), replace=False)
    train_id = np.array([i for i in files if i not in validate_id])
    
    return (trainset.loc[trainset['file'].isin(train_id)].reset_index(drop=True),
            trainset.loc[trainset['file'].isin(validate_id)].reset_index(drop=True))
  
  train, validate = split(trainset)
  np.random.seed(123)
  subset = np.random.choice(sorted(train['file'].unique()), subset_n, replace=False)
  train = train.loc[train['file'].isin(subset)]
  """ load pre-trained models """
  tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

  """ seq input """
  def seq_input(seq):
    return tokenizer.batch_encode_plus(seq, 
                              max_length=max_length, 
                              pad_to_max_length=True, 
                              return_tensors="tf",
                              return_attention_mask=False,
                              return_token_type_ids=False)['input_ids']
  
  train_seq = seq_input(train['hor_seq'])
  validate_seq = seq_input(validate['hor_seq'])
  test_seq = seq_input(test['hor_seq'])
  
  """ fixed input """ 
  var = ['left', 'top', 'width', 'height', 'page', 'num']
  train_fix = np.array(train[var])
  validate_fix = np.array(validate[var])
  test_fix = np.array(test[var])
  
  """ make label """
  train_label = np.array(pd.get_dummies(train, columns=['label'])[['label_AHI', 'label_SaO2', 'label_Other']])
  validate_label = np.array(pd.get_dummies(validate, columns=['label'])[['label_AHI', 'label_SaO2', 'label_Other']])
  test_label = np.array(pd.get_dummies(test, columns=['label'])[['label_AHI', 'label_SaO2', 'label_Other']])
  
  """ metrics """
  def AHI_recall(y_true, y_pred):
    label = K.cast(y_true[:,0], 'float32')
    pred = K.cast((y_pred[:,0]>y_pred[:,1]) & (y_pred[:,0]>y_pred[:,2]), 'float32')
    true_positives = K.sum(K.round(label * pred))
    possible_positives = K.sum(label)
    return true_positives / (possible_positives + K.epsilon())
  
  def AHI_precision(y_true, y_pred):
    label = K.cast(y_true[:,0], 'float32')
    pred = K.cast((y_pred[:,0]>y_pred[:,1]) & (y_pred[:,0]>y_pred[:,2]), 'float32')
    true_positives = K.sum(K.round(label * pred))
    predicted_positives = K.sum(pred)
    return true_positives / (predicted_positives + K.epsilon())
  
  def AHI_f1(y_true, y_pred):
    recall = AHI_recall(y_true, y_pred)
    prec = AHI_precision(y_true, y_pred)
    return 2*recall*prec / (recall + prec + K.epsilon())
  
  """ model """
  def model_structure(model_dir):
    bert = TFBertModel.from_pretrained(model_dir, from_pt=True)
    """ seq branch """
    input_ids = layers.Input(shape=(max_length,), name='BERT_input', dtype=tf.int32)
    embedding = bert(input_ids)[0]
    Flatten_layer = layers.Flatten(name='Flatten_layer')(embedding)
    summary_seq = layers.Dense(200, activation="relu", name='Summarys_seq')(Flatten_layer)
    seq_branch = tf.keras.Model(inputs=input_ids, outputs=summary_seq)
    """ fix branch """
    input_fix = layers.Input(shape=(6,), name='Fix_input')
    norm_fix = layers.BatchNormalization()(input_fix)
    dense_fix1 = layers.Dense(100, activation="relu", name = 'dense_fix1')(norm_fix)
    dropout_fix1 = layers.Dropout(0.2, name='dropout_fix1')(dense_fix1)
    dense_fix2 = layers.Dense(100, activation="relu", name = 'dense_fix2')(dropout_fix1)
    dropout_fix2 = layers.Dropout(0.2, name='dropout_fix2')(dense_fix2)
    fix_branch = tf.keras.Model(inputs=input_fix, outputs=dropout_fix2)
    """ combine """
    combined = layers.concatenate([seq_branch.output, fix_branch.output], name='Conbine')
    summary = layers.Dense(200, activation="relu", name='Summary')(combined)
    dropout_summary = layers.Dropout(0.2, name='dropout_summary')(summary)
    """ output """
    out = layers.Dense(3, activation="sigmoid", name='out')(dropout_summary)
    """ define model """
    model = tf.keras.Model(inputs=[seq_branch.input, fix_branch.input], outputs=out)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(lr=2e-6), 
                  metrics=[AHI_f1, AHI_recall, AHI_precision])
    return model
  
  """ define model """
  model = model_structure(ClinicalBERT_dir)
  
  #plot_model(model, show_shapes = False)
  #model.summary()
  
  """ fine-tune BERT """
  filepath = f'{model_dir}\\' + "epoch-{epoch}-loss-{loss:.4f}-val_loss-{val_loss:.4f}-val_AHI_F1-{val_AHI_f1:.4f}-val_AHI_recall-{val_AHI_recall:.4f}-val-AHI_precision-{val_AHI_precision:.4f}.h5"
  callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
  model.fit([train_seq, train_fix], train_label, validation_data = ([validate_seq, validate_fix], validate_label), epochs=100, batch_size=64, callbacks=[callback])
  
  """ output pred """
  models = sorted([m for m in listdir(f'{model_dir}') if isfile(join(f'{model_dir}', m))])
  best = model_structure(ClinicalBERT_dir)
  best.load_weights(f'{model_dir}\\{models[-1]}')
  pred = best.predict([test_seq, test_fix])
  prediction = pd.DataFrame(pred, columns=['AHI', 'SaO2', 'Other'])
  prediction['pred'] = prediction.idxmax(axis=1)
  prediction['prob_AHI'] = prediction['AHI']/prediction[['AHI', 'SaO2', 'Other']].sum(axis=1)
  prediction['prob_SaO2'] = prediction['SaO2']/prediction[['AHI', 'SaO2', 'Other']].sum(axis=1)
  prediction['label'] = list(test['label'])
  prediction['pred_AHI'] = prediction['pred'] == 'AHI'
  prediction['label_AHI'] = prediction['label'] == 'AHI'
  prediction['pred_SaO2'] = prediction['pred'] == 'SaO2'
  prediction['label_SaO2'] = prediction['label'] == 'SaO2'
  
  prediction.to_pickle(f'{pred_file}.pickle')
  print('***ClinicalBERT finished without errors***')
