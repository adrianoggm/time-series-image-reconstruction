
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical

""" Generates and saves a time series from a RP saving keeping also 
    track of the acuracy and the metrics from the original Time series
"""
def generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_data, sj_train, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))
    
    for i in p_bar:
      w = training_data[i]
      sj = sj_train[i][0]
      w_y = y_data[i]
      w_y_no_cat = np.argmax(w_y)
      print("w_y", w_y, "w_y_no_cat", w_y_no_cat)
      print("w", w.shape)
     
      # Update Progress Bar after a while
      time.sleep(0.01)
      p_bar.set_description(f'[{data_type} | FOLD {fold} | Class {w_y_no_cat}] Subject {sj}')
      path=f"{path}{sj}{"x"}{subject_samples}.png"  
      imagen = cv2.imread(path)  
      imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
      ##We need to change path 
      path=f"{dataset_folder}tseries/recurrence_plot_aqui tiene que cambiarla vaina/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
      rp=Reconstruct_RP(imagen,path)
      error=ts_error(w,rp)
      subject_samples += 1
##Calculates all different error measures for each channel returning them in n-channel tuples   
def ts_error(original,creada):
    errores_absolutos=[]
    errores_relativos=[]
    errores_d=[]
    errores_pearson =[]
    for i in range(0,3):
      error_absoluto, error_relativo = calcular_errores(original[i], creada[i])
      d = dtw.distance_fast(original[i], creada[i], use_pruning=True)
      pearson=np.corrcoef(original[i], creada[i])[0,1]
      print(f"Error Absoluto Promedio: {error_absoluto}")
      print(f"Error Relativo Promedio: {error_relativo}")
      print(f"Error DTW: {d}")
      print(f"Coeficiente de correlación: {pearson}")
      np.append(errores_absolutos,error_absoluto)
      np.append(errores_relativos,error_relativo)
      np.append(errores_d,d)
      np.append(errores_pearson,pearson)
    return  errores_absolutos, errores_relativos,errores_d,errores_pearson   


##Calculate absolute error and  relative error
def calcular_errores(valores_verdaderos, valores_aproximados):
    # Convertir las listas a arrays de numpy para facilitar los cálculos
    valores_verdaderos = np.array(valores_verdaderos)
    valores_aproximados = np.array(valores_aproximados)
    
    # Calcular el error absoluto
    errores_absolutos = np.abs(valores_verdaderos - valores_aproximados)
    
    # Calcular el error relativo (evitando la división por cero)
    errores_relativos = np.abs(errores_absolutos / valores_verdaderos)
    
    # Calcular el error absoluto promedio
    error_absoluto_promedio = np.mean(errores_absolutos)
    
    # Calcular el error relativo promedio
    error_relativo_promedio = np.mean(errores_relativos)
    
    return error_absoluto_promedio, error_relativo_promedio

#Generates all time series
def generate_all_time_series(X_train, y_train, sj_train, dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", TIME_STEPS=129,  FOLDS_N=3, sampling="loso"):
  groups = sj_train 
  if sampling == "loto":
    #TODO change 100 for an automatic extracted number greater than the max subject ID: max(sj_train)*10
    groups = [[int(sj[0])+i*100+np.argmax(y_train[i])+1] for i,sj in enumerate(sj_train)]

  # if DATASET_NAME == "WISDM": #since wisdm is quite balanced
  sgkf = StratifiedGroupKFold(n_splits=FOLDS_N)
  # elif DATASET_NAME == "MINDER" or DATASET_NAME == "ORIGINAL_WISDM": 
  #   sgkf = StratifiedGroupKFold(n_splits=FOLDS_N)

  accs = []
  y_train_no_cat = [np.argmax(y) for y in y_train]
  p_bar_classes = tqdm(range(len(np.unique(y_train_no_cat))))
  all_classes = np.unique(y_train_no_cat)
  print("Classes available: ", all_classes)
  for fold in range(FOLDS_N):
    for i in p_bar_classes:
        y = all_classes[i]
        time.sleep(0.01) # Update Progress Bar after a while
        os.makedirs(f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
        os.makedirs(f"{dataset_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)

  for fold, (train_index, val_index) in enumerate(sgkf.split(X_train, y_train_no_cat, groups=groups)):

    # if fold != 2:
    #   continue

    # print(f"{'*'*20}\nFold: {fold}\n{'*'*20}")
    # print("Train index", train_index)
    # print("Validation index", val_index)
    training_data = X_train[train_index,:,:]
    validation_data = X_train[val_index,:,:]
    y_training_data = y_train[train_index]
    y_validation_data = y_train[val_index]
    sj_training_data = sj_train[train_index]
    sj_validation_data = sj_train[val_index]

    print("training_data.shape", training_data.shape, "y_training_data.shape", y_training_data.shape, "sj_training_data.shape", sj_training_data.shape)
    print("validation_data.shape", validation_data.shape, "y_validation_data.shape", y_validation_data.shape, "sj_validation_data.shape", sj_validation_data.shape)


    ##aqui añadir una opcion si quieres todas o una en especifico
    generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_training_data, sj_training_data, TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    generate_and_save_time_series_fromRP(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)