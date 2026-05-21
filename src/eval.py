# Módulos incorporados (built-in)
import os
import time
import math
import json
import pathlib
from pathlib import Path
import copy
from copy import deepcopy

# Librerías de terceros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")  # Estilo para Seaborn

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from PIL import Image

# Barras de progreso
from tqdm import tqdm, trange
from tqdm.notebook import tqdm as notebook_tqdm  # Si se requiere en notebooks

# PyTorch y complementos
import torch
import torch.nn.functional as TF
import torchvision
from torchvision import datasets, transforms, utils as tv_utils
from torch.utils.data import DataLoader, ConcatDataset
from torch import nn, optim
from torch.utils import data
from torch import multiprocessing as mp

# Carga de datasets (por ejemplo, Hugging Face)
from datasets import load_dataset

# Keras
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import backend as K


sns.set_style("darkgrid")



def plot_cm(dir_path, model_name, y_true, y_pred, labels, normalized=True, figsize=(3, 4), suffix_name="", title='Confusion Matrix', cmap=plt.cm.Blues, fontsize=16):
  '''
    Plots a confusion matrix and saves the metrics tp, tn, fp, fn, accuracy, sensitivity, specificity as a pandas DataFrame.

    Parameters:
    y_true (array-like): True labels
    y_pred (array-like): Predicted labels
    labels (array-like): List of labels for each class
    normalize (bool): Whether to normalize the confusion matrix or not (default False)
    title (str): Title of the plot (default 'Confusion Matrix')
    cmap (matplotlib colormap): Colormap for the plot (default plt.cm.Blues)

    Returns:
    pandas DataFrame with the following metrics:
    tp (int): True positives
    tn (int): True negatives
    fp (int): False positives
    fn (int): False negatives
    accuracy (float): Accuracy
    sensitivity (float): Sensitivity (True Positive Rate)
    specificity (float): Specificity (True Negative Rate)
  '''
  classes = list(labels)
  # n_labels = len(labels)
  # Compute confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  
  # Compute normalized confusion matrix
  cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  # Get true positives, true negatives, false positives, false negatives
  tp = np.diag(cm)
  fp = np.sum(cm, axis=0) - tp
  fn = np.sum(cm, axis=1) - tp
  tn = np.sum(cm) - (tp + fp + fn)

  # Compute accuracy, sensitivity, and specificity
  accuracy = (tp.sum() + tn.sum()) / (tp.sum() + fp.sum() + tn.sum() + fn.sum())
  sensitivity = tp / (tp + fn)
  specificity = tn / (tn + fp)

  # Create dataframe with metrics
  metrics_df = pd.DataFrame({'Accuracy': [accuracy],
                              'Sensitivity': [sensitivity],
                              'Specificity': [specificity]})
  print({'Accuracy': [accuracy],
                              'Sensitivity': [sensitivity],
                              'Specificity': [specificity]})


  # Print metrics
  print(metrics_df)

  # Plot confusion matrix
  fig, ax = plt.subplots(figsize=figsize)
  sns.heatmap(cm, annot=True, cmap=cmap, square=True, ax=ax,
              xticklabels=labels, yticklabels=labels, fmt='g', annot_kws={"fontsize": fontsize})
  ax.set_xlabel('Predicted label',fontsize=fontsize)
  ax.set_ylabel('True label',fontsize=fontsize)
  ax.set_title('Confusion Matrix',fontsize=fontsize)
  ax.set_xticklabels(classes, rotation=45, ha='right',fontsize=fontsize)
  ax.set_yticklabels(classes, rotation=0,fontsize=fontsize)
  
  # Save confusion matrix plot
  plt.savefig(f"{dir_path}cm.png", dpi=300, bbox_inches='tight')
  
  # Plot normalized confusion matrix
  fig, ax = plt.subplots(figsize=figsize)
  sns.heatmap(cm_norm, annot=True, cmap=cmap, square=True, ax=ax,
              xticklabels=labels, yticklabels=labels, fmt='.3f', annot_kws={"fontsize": fontsize})
  ax.set_xlabel('Predicted label',fontsize=fontsize)
  ax.set_ylabel('True label',fontsize=fontsize)
  ax.set_title('Normalized Confusion Matrix',fontsize=fontsize)
  ax.set_xticklabels(classes, rotation=45, ha='right',fontsize=fontsize)
  ax.set_yticklabels(classes, rotation=0,fontsize=fontsize)

  # Save normalized confusion matrix plot
  plt.savefig(f"{dir_path}norm_cm.png", dpi=300, bbox_inches='tight')
  
  # Save metrics to CSV
  metrics_df.to_csv(f"{dir_path}cm_metrics.csv", index=False)
  return metrics_df

# cm = confusion_matrix(y_true, y_pred)
# tp = np.diag(cm)
# fp = []
# for i in range(n_labels):
#   fp.append(sum(cm[:,i]) - cm[i,i])
# fn = []
# for i in range(n_labels):
#   fn.append(sum(cm[i,:]) - cm[i,i])
# tn = []
# for i in range(n_labels):
#   temp = np.delete(cm, i, 0)   # delete ith row
#   temp = np.delete(temp, i, 1)  # delete ith column
#   tn.append(sum(sum(temp)))
# # Let's make a sanity check: for each class, the sum of TP, FP, FN, and TN must be equal to the size of our test set 
# l = len(y_pred)
# for i in range(n_labels):
#     print(f"the sum of TP, FP, FN, and TN must be equal to the size of our test set [for class: {i}] ")
#     print(tp[i] + fp[i] + fn[i] + tn[i] == l)
# total_samples = sum(tp) + sum(fp) + sum(fn) + sum(tn)
# acc = (sum(tp)+sum(tn)) / total_samples
# miss = (sum(fp)+sum(fn)) / total_samples
# tpr = sum(tp) / (sum(tp) + sum(fp))
# tnr = sum(tn) / (sum(tn) + sum(fn))
# tpr_prod_tnr = tpr*tnr
# print("Accuracy:", acc, "Missclassification:", miss, "TPR:", tpr, "TNR:", tnr, "TPR*TNR:", tpr_prod_tnr)
# pd.DataFrame([[acc, miss, tpr, tnr, tpr_prod_tnr, sum(tp), sum(tn), sum(fp), sum(fn)]], columns=["Accuracy", "Missclassification", "TPR", "TNR", "TPR*TNR", "TP", "TN", "FP", "FN"]).to_csv(f"{dir_path}/cm_results.csv",index=False)

# if normalized:
#   normalized_cm = np.zeros((n_labels, n_labels))
#   for i in range(n_labels):
#       for j in range(n_labels):
#           normalized_cm[i][j] = float(cm[i][j])/sum(cm[i])
#   cm = normalized_cm

# diag = np.diag(cm)

# fig, ax = plt.subplots(figsize=figsize) 
# ax = sns.heatmap(
#     cm, 
#     annot=True, 
#     # fmt="d", 
#     cmap=sns.diverging_palette(220, 20, n=7),
#     ax=ax
# )

# plt.title(f'Confussion Matrix for Sythetic data')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# ax.set_xticklabels(class_names)
# ax.set_yticklabels(class_names)
# b, t = plt.ylim() # discover the values for bottom and top
# b += 0.5 # Add 0.5 to the bottom
# t -= 0.5 # Subtract 0.5 from the top
# plt.ylim(b, t) # update the ylim(bottom, top) values
# p = f"{dir_path}/cm_{model_name}_{suffix_name}.png"
# plt.savefig(p, dpi=300, bbox_inches='tight')
# plt.show() # ta-da!

def plot_acc_bar(filename, leg_title, accs, perc_augmentation, plot_type="bar", stds_fill={}, figsize=(12, 8), save_path_dir="", legend_out=False, title=""):  
  if type(accs) == pd.core.frame.DataFrame:
    plotdata = accs
    plotdata = plotdata.set_index(pd.Series(perc_augmentation))
  else:
    plotdata = pd.DataFrame(
        accs 
        # a dict with this shape
        # {
        # "cnn_a":[0.88, 0.89, 0.90, 0.92, 0.99], #same len of histories
        # "cnn_b":[0.78, 0.85, 0.90, 0.92, 0.99], #same len of histories
        # "cnn_c":[0.68, 0.85, 0.90, 0.92, 0.99] #same len of histories
        # }
      , index=perc_augmentation #same len of histories
    )
  # p = f"{SOURCE_PATH}/models/DDPM-classif/{MODEL_NAME}/best_model_kfold_classes_accs.csv"
  # plotdata.to_csv(p)
  
  fig, ax = plt.subplots(figsize=figsize) 
  sns.set_style("dark")


  #1. Accuracy lines (this order matters)
  plotdata.plot(kind=plot_type, rot=0, ax=ax)

  #2. stds line (this order matters)
  if len(stds_fill) > 0:
    for k in stds_fill.keys():
      print("Filling")
      plt.fill_between(perc_augmentation, 
                       np.array(accs[k]) - np.array(stds_fill[k]), 
                       np.array(accs[k]) + np.array(stds_fill[k]), alpha=.1)


  
  plt.tight_layout()
  plt.title(title)
  plt.xlabel("Synthetic Data Percentages")
  ax.legend(title=leg_title) #, loc='center left', bbox_to_anchor=(1, 0.5))
  if legend_out:
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.grid()

  m="Validation Accuracy"
  plt.ylabel(f"{m} Score")
  # b, t = plt.ylim() # discover the values for bottom and top
  # b += 0.5 # Add 0.5 to the bottom
  # t -= 0.5 # Subtract 0.5 from the top
  # plt.ylim(b, t) # update the ylim(bottom, top) values
  
  if save_path_dir != "":
    p = f"{save_path_dir}/{plot_type}_{MODEL_NAME}_{filename}.png"
  else:
    p = f"{SOURCE_PATH}/models/DDPM-classif/{MODEL_NAME}/{plot_type}_{MODEL_NAME}_{filename}.png"
  plt.savefig(p, dpi=300, bbox_inches='tight')
  plt.show()

#@title Different CNN model architectures
def compile(model):

  # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
  #   initial_learning_rate=0.001, #1e-2,
  #   decay_steps=10000,
  #   decay_rate=0.9)
  # opt = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
  opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)

  model.compile(optimizer=opt,
  loss="categorical_crossentropy",
  metrics=["accuracy"])
  return model
# model.compile(
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       optimizer = optimizers.Adam(),
#       metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
#   )


def fit(dir_path, name, kfold, model, train_ds, val_ds, epochs, batch_size, device=-1, patience=20, printsummary=False):
  # model = compile(model)
  # if printsummary:
  #   print(model.summary())

  print("setup")

  # earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=patience, verbose=1, mode='max')
  p = f'{dir_path}fold_{kfold}/{name}/model_{name}'
  os.makedirs(p, exist_ok=True)

  #without specifying .hdf5 for using .tf by default (if we are using Customs models)
  monitor = 'val_accuracy' if val_ds is not None else 'loss'
  model_checkpoint = keras.callbacks.ModelCheckpoint(p, save_best_only=True, monitor=monitor, mode='max', verbose=1)
  # reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


  # early_stopping = keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)

  # history = model.fit(train_ds, 
  #   # steps_per_epoch=train_ds.samples // batch_size ,  #  images / batch_size = steps
  #   epochs=epochs, 
  #   validation_data=val_ds, 
  #   # validation_steps= val_ds.samples // batch_size,
  #   batch_size = BATCH_SIZE,
  #   callbacks= [earlyStopping, mcp_save] if USE_EARLY_STOP else [mcp_save], #[early_stopping],
  #   verbose=2
  # )

  # learning_rate=1e-3
  # factor = 1. / np.sqrt(2)
  # reduce_lr =  keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=100, mode='auto',
  #                               factor=factor, cooldown=0, min_lr=1e-4, verbose=2)
  # callback_list = [model_checkpoint, reduce_lr]

  # opt =  keras.optimizers.SGD(lr=learning_rate, momentum = 0.9)
  opt = "adam"
  callback_list = [model_checkpoint]
  # opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


  #create model and train
  # model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


  if printsummary:
    print(model.summary())

  STEP_SIZE_TRAIN=train_ds.n//train_ds.batch_size
  # if device >= 0:
  #   with tf.device(f'/device:GPU:{device}'):
  #     history = model.fit(train_ds,
  #                         steps_per_epoch=STEP_SIZE_TRAIN,
  #                         validation_data=val_ds,
  #                         validation_steps=STEP_SIZE_VALID,
  #                         callbacks=callback_list,
  #                         epochs=epochs
  #     )
  # else:
  if val_ds is None:
    history = model.fit(train_ds,
                          steps_per_epoch=STEP_SIZE_TRAIN,
                          callbacks=callback_list,
                          epochs=epochs,
                          verbose=1
    )
  else:
    STEP_SIZE_VALID=val_ds.n//val_ds.batch_size

    history = model.fit(train_ds,
                          steps_per_epoch=STEP_SIZE_TRAIN,
                          validation_data=val_ds,
                          validation_steps=STEP_SIZE_VALID,
                          callbacks=callback_list,
                          epochs=epochs,
                          verbose=1
      )


  # if device >= 0:
  #   with tf.device(f'/device:GPU:{device}'):
  #     history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=2, validation_data=val_ds)
  # else:
  #   history = model.fit(train_ds, batch_size=batch_size, epochs=epochs, callbacks=callback_list, verbose=2, validation_data=val_ds)



  return model, history

def generate_lstmfcn_xia(dir_path, model_name, kfold, n_labels, input_shape, train_ds, val_ds, epochs, batch_size, device=-1, patience=10, printsummary=True):
  '''https://ieeexplore.ieee.org/document/9043535/'''

  ip = keras.layers.Input(shape=input_shape)
  x = keras.layers.Reshape((input_shape[0]*input_shape[1], input_shape[2]), input_shape=input_shape)(ip)
  x = keras.layers.LSTM(32, return_sequences=True)(x)
  x = keras.layers.LSTM(32, return_sequences=True)(x)

    
  x = keras.layers.Conv1D(64, kernel_size=5, strides=2, padding='same')(x)
  x = keras.layers.MaxPooling1D(pool_size=2, strides=2)(x)

  x = keras.layers.Conv1D(128, kernel_size=3, strides=1, padding='same')(x)

  x = keras.layers.GlobalAveragePooling1D()(x)
  x = keras.layers.BatchNormalization()(x)

  out = keras.layers.Dense(n_labels, activation='softmax')(x)

  model = keras.models.Model(ip, out)

  print(model.summary())

  return fit(dir_path, model_name, kfold, model, train_ds, val_ds, epochs, batch_size, device, patience, printsummary)


def build_X(dir_path, model_name, kfold, n_labels, input_shape, train_ds, val_ds, epochs, batch_size, device=-1, patience=10, printsummary=True):
  # model = ResNet18(n_labels).model(input_shape=input_shape)
  from classification_models.keras import Classifiers
  model_get, preprocess_input = Classifiers.get(model_name)
  print("input:", preprocess_input)
  print("model:", model_get)

  base_model = model_get(input_shape=input_shape, weights='imagenet', include_top=False)
  # # freeze base layers
  # for layer in base_model.layers:
  #     layer.trainable=False

  print("loaded base model")
  x = keras.layers.GlobalAveragePooling2D()(base_model.output)
  output = keras.layers.Dense(n_labels, activation='softmax')(x)
  model = keras.models.Model(inputs=[base_model.input], outputs=[output])
  return fit(dir_path, model_name, kfold, model, train_ds, val_ds, epochs, batch_size, device, patience, printsummary)


# def build_tiny_resnet(perc, n_labels, input_shape, train_ds, val_ds, epochs, patience=10, printsummary=True):
#   model = TinyResNet(n_labels).model(input_shape=input_shape)
#   # model.build(input_shape=input_shape)
#   return fit("tiny_resnet", perc, model, train_ds, val_ds, epochs, patience, printsummary)

# def build_model_FSNet(perc, n_labels, train_ds, val_ds, epochs, patience=10, printsummary=True):
#   model = keras.models.Sequential([
    
#     #Convolution 3x3
#     keras.layers.Conv2D(32, kernel_size=(3,3), strides=1, activation="relu", input_shape=(IMG_SIZE,IMG_SIZE, 3)),
#     keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
    
#     #Max pooling
#     keras.layers.MaxPooling2D((2,2)),

    
#     #residual module 1
#     keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
#     keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
#     keras.layers.Conv2D(64, kernel_size=3, strides=1, activation="relu"),
#     keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

#     #residual module 2
#     keras.layers.Conv2D(128, kernel_size=3, strides=1, activation="relu"),
#     keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
#     keras.layers.Conv2D(128, kernel_size=3, strides=1, activation="relu"),
#     keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

#     #Max pooling
#     keras.layers.MaxPooling2D((2,2)),

#      #residual module 3
#     # keras.layers.Conv2D(32, kernel_size=3, strides=1, activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
#     # keras.layers.Conv2D(32, kernel_size=3, strides=1, activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

#     # #residual module 4
#     # keras.layers.Conv2D(32, (3,3), activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
#     # keras.layers.Conv2D(32, (3,3), activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

#     # #Max pooling
#     # keras.layers.MaxPooling2D((2,2)),

#     #  #residual module 5
#     # keras.layers.Conv2D(32, (3,3), activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
#     # keras.layers.Conv2D(32, (3,3), activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

#     # #residual module 6
#     # keras.layers.Conv2D(32, (3,3), activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
#     # keras.layers.Conv2D(32, (3,3), activation="relu"),
#     # keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

#     # #Max pooling
#     # keras.layers.MaxPooling2D((2,2)),

#     #Global
#     keras.layers.GlobalAveragePooling2D(),

#     #Fully Connected Layer
#     keras.layers.Dense(n_labels, activation="softmax")                         
#   ])


#   return fit("fsnet", perc, model, train_ds, val_ds, epochs, patience, printsummary)



def build_model_A(dir_path, model_name, kfold, n_labels, input_shape, train_ds, val_ds, epochs, batch_size, device=-1, patience=10, printsummary=True):
  model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape),
    keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

    keras.layers.MaxPooling2D((2,2)),
    
    keras.layers.Conv2D(64, (3,3), activation="relu"),
    keras.layers.BatchNormalization(), #applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.

    keras.layers.MaxPooling2D((2,2)),
    
    # layers.Flatten(),
    # layers.Dense(512, activation="relu"),

    keras.layers.GlobalAveragePooling2D(),

    
    keras.layers.Dense(n_labels, activation="softmax")                         
  ])
  return fit(dir_path, model_name, kfold, model, train_ds, val_ds, epochs, batch_size, device, patience, printsummary)


# def build_model_B(perc, n_labels, train_ds, val_ds, epochs, patience=10, printsummary=True):
#   model = keras.models.Sequential([
#     keras.layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE,IMG_SIZE, 3)),
#     keras.layers.MaxPooling2D((2,2)),
#     keras.layers.Conv2D(32, (3,3), activation="relu"),
#     keras.layers.MaxPooling2D((2,2)),
#     keras.layers.Conv2D(64, (3,3), activation="relu"),
#     keras.layers.MaxPooling2D((2,2)),
#     keras.layers.Conv2D(64, (3,3), activation="relu"),
#     keras.layers.MaxPooling2D((2,2)),
#     keras.layers.Flatten(),
#     keras.layers.Dense(512, activation="relu"),
#     keras.layers.Dense(n_labels, activation="softmax")                         
#   ])
#   return fit("b", perc, model, train_ds, val_ds, epochs, patience, printsummary)

# def build_model_C(perc, n_labels, train_ds, val_ds, epochs, patience=10, printsummary=True):
#   model = keras.models.Sequential([
#   keras.layers.Conv2D(image_size, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#   keras.layers.MaxPooling2D((2, 2)),
#   keras.layers.Conv2D(image_size*2, (3, 3), activation='relu'),
#   keras.layers.MaxPooling2D((2, 2)),
#   keras.layers.Conv2D(image_size*2, (3, 3), activation='relu'),
#   keras.layers.Flatten(),
#   keras.layers.Dense(image_size*2, activation='relu'),
#   keras.layers.Dense(n_labels, activation="softmax")
#   ])
#   return fit("c", perc, model, train_ds, val_ds, epochs, patience, printsummary)

# def build_model_finetue(base_model, n_freezes, perc, n_labels, train_ds, val_ds, epochs, patience=10, printsummary=True):
#   # base_model.trainable = False
#   #Freeze layers
#   for layer in base_model.layers[:n_freezes]:
#     layer.trainable = False
#   #Check we correctly freezed the desired layers
#   # for i,layer in enumerate(base_model.layers):
#   #   print(i, layer.name, "-", layer.trainable)
  
#   #We can use global pooling or a flatten layer to connect the dimensions of the previous layers with the new layers. 

#   model = tf.keras.models.Sequential([base_model])
#   model.add(tf.keras.layers.GlobalAvgPool2D())
#   # model.add(tf.keras.layers.Dense(128,activation='relu'))
#   # model.add(tf.keras.layers.Dense(n_labels, activation = 'softmax'))  
#   model.add(keras.layers.Dense(n_labels, activation="softmax"))
#   return fit("fine_tune", perc, model, train_ds, val_ds, epochs, patience, printsummary)

def get_metrics(history):
  history = history.history
  acc = history['accuracy']
  val_acc = history['val_accuracy']
  loss = history['loss']
  val_loss = history['val_loss']
  return acc, val_acc, loss, val_loss

def plot_train_eval(history, title):
  acc, val_acc, loss, val_loss = get_metrics(history)

  plt.figure(figsize=(12, 4))
  ax = plt.subplot(1, 2, 1)
  acc_plot = pd.DataFrame({"training accuracy":acc, "evaluation accuracy":val_acc})
  acc_plot = sns.lineplot(data=acc_plot)
  acc_plot.set_title(f'{title}')
  acc_plot.set_xlabel('epoch')
  acc_plot.set_ylabel('accuracy')

  ax = plt.subplot(1, 2, 2)
  loss_plot = pd.DataFrame({"training loss":loss, "evaluation loss":val_loss})
  loss_plot = sns.lineplot(data=loss_plot)
  loss_plot.set_title(f'{title}')
  loss_plot.set_xlabel('epoch')
  loss_plot.set_ylabel('loss')
  plt.show()

def plot_train_synthetic_data_perc(histories, vis_metric="val_acc", perc_augmentation=[0, 0.1]):
  '''TODO fix to be more flexible, histories coud be a dictioniary where keys
  were the model_name and values a list of resultf for every %of augmented
  synthetic data and the metrics. Example:
  "{
    cnn_a":[0% >> {acc:[every epoch], val_acc:[every epoch]}, [next 0.1% >> ...]],
    cnn_b":[0% >> {acc:[every epoch], val_acc:[every epoch]}, [next 0.1% >> ...]]
    ... '''
  accs, val_accs, losses, val_losses = [], [], [], []

  for h in histories:
    acc, val_acc, loss, val_loss = get_metrics(h)
    accs.append(np.max(acc))
    val_accs.append(np.max(val_acc))
    losses.append(np.min(loss))
    val_losses.append(np.min(val_loss))
  
  data_vis = {"cnn_a": val_accs}
  if vis_metric == "acc":
    data_vis = {"cnn_a": accs}

  plotdata = pd.DataFrame(
      data_vis
      # {
      # "cnn_a":[0.88, 0.89, 0.90, 0.92, 0.99], #same len of histories
      # "cnn_b":[0.78, 0.85, 0.90, 0.92, 0.99], #same len of histories
      # "cnn_c":[0.68, 0.85, 0.90, 0.92, 0.99] #same len of histories
      # }
    , index=perc_augmentation #same len of histories
  )
  sns.set_style("dark")
  plotdata.plot(kind="bar", rot=0)
  plt.tight_layout()
  plt.title("Classification performance of various CNN architectures")
  plt.xlabel("Synthetic Data Percentages")

  m="Validation Accuracy"
  if vis_metric == "acc":
    m = "Accuracy"
  elif vis_metric == "loss":
    m = "Loss"
  elif vis_metric == "val_loss":
    m = "Validation Loss"
  plt.ylabel(f"{m} Score")


    
        