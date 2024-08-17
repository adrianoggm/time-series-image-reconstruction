
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from math import sqrt
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField
from scipy.sparse.csgraph import dijkstra
from PIL import Image
import cv2
from sklearn.manifold import MDS
from dtaidistance import dtw
from matplotlib.pylab import rcParams
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
#import calculate_ts_errors as err
import tgen.calculate_ts_errors as err
import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import tgen.recurrence_plots as rpts
import tgen.GAF as gaf
#import tgen.GAF2 as gaf2
import time
import tgen.MTF as mtf2
import os    

""" Generates and saves a time series from a RP saving keeping also 
    track of the acuracy and the metrics from the original Time series
"""
def generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_data, sj_train,dictionary, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
    
    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))
    errores=[]
    X_all_rec=[]
    tiempos=[]
    start_gl=time.time()
    #Colocar timer
    ninv=0
    
    for i in p_bar:
      #start=time.time()
      w = training_data[i]
      sj = sj_train[i][0]
      w_y = y_data[i]
      w_y_no_cat = np.argmax(w_y)
      #print("w_y", w_y, "w_y_no_cat", w_y_no_cat)
      #print("w", w.shape)
      error=[]
      # Update Progress Bar after a while
      #time.sleep(0.01)
      #p_bar.set_description(f'[{data_type} | FOLD {fold} | Class {w_y_no_cat}] Subject {sj}')
      path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
      path=f"{path}{sj}x{subject_samples}.png"  
      imagen = cv2.imread(path)  
      imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
      ##We need to change path 
      rp,a=rpts.Reconstruct_RP(imagen,dictionary,subject_samples)
      #end=time.time()
      ninv+=a
      #Si queremos guardar los datos
      path=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/{sj}x{subject_samples}.npy"
      np.save(path, np.array(rp))
      
      #print(rp)
      X_all_rec.append(rp)
      #print("X_ALL",np.array(X_all_rec))
      #tiempo=end-start
      #tiempos.append(tiempo)
      #Guardar la rp en el path indicado con un nombre adecuado
      #print(tiempos)
      #w=w.reshape(3,129)
      #w=w[:, 0]
      #w=w[1:]
      #print("Forma del RP original y calculada",w.shape,rp.shape)
      
      #print("normal",w,"calculada",rp)
      """
      error_abs,error_r,error_q,error_std,error_p,te=err.ts_error(w,rp)
      t+=te
      error.append(error_abs)
      error.append(error_r)
      error.append(error_q)
      error.append(error_std)
      error.append(error_p)
      errores.append(error)
      #print("errores",np.array(error))
      #print("errores",np.array(errores))
      """
      subject_samples += 1

    end_gl=time.time()
    ttotal=end_gl-start_gl

    #maybe we should calculate here the global of errors and the mean.
    archivoX_all=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/X_all_rec.npy"
    np.save(archivoX_all,np.array(X_all_rec))
    #archivoerrores=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/errores_rec.npy"
    #np.save(archivoerrores,np.array(errores))

    #archivotiempos=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/tiempos_rec.npy"
    #np.save(archivotiempos,np.array(ttotal/subject_samples))
    #print("Tiempo medio",np.mean(tiempos))
    print("Tiempo medio total",np.mean(ttotal/subject_samples))
    print("Número total de inversiones",ninv)
    #print("Numero total desplazamientos",t)
    return X_all_rec,errores


def generate_and_save_time_series_fromGAF(fold, dataset_folder, training_data, y_data, sj_train,dictionary, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):

    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))
    X_all_rec=[]
    tiempos=[]
    start_gl=time.time()
    
    #Colocar timer
    
    
    for i in p_bar:
      #start=time.time()
      w = training_data[i]
      sj = sj_train[i][0]
      w_y = y_data[i]
      w_y_no_cat = np.argmax(w_y)
      #print("w_y", w_y, "w_y_no_cat", w_y_no_cat)
      #print("w", w.shape)
      
      # Update Progress Bar after a while
      #time.sleep(0.01)
      #p_bar.set_description(f'[{data_type} | FOLD {fold} | Class {w_y_no_cat}] Subject {sj}')
      path=f"{dataset_folder}plots/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
      path=f"{path}{sj}x{subject_samples}gasf.png"  
      imagen = cv2.imread(path)  
      imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
      ##We need to change path 
      gasf=gaf.Reconstruct_GAF(imagen,dictionary,subject_samples)
      #end=time.time()
      
      #Si queremos guardar los datos
      path=f"{dataset_folder}tseries/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/{sj}x{subject_samples}.npy"
      np.save(path, np.array(gasf))
      
      #print(rp)
      X_all_rec.append(gasf)
   
      
      subject_samples += 1

    end_gl=time.time()
    ttotal=end_gl-start_gl

    #maybe we should calculate here the global of errors and the mean.
    archivoX_all=f"{dataset_folder}tseries/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/X_all_gaf.npy"
    np.save(archivoX_all,np.array(X_all_rec))
    

    #archivotiempos=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/tiempos_rec.npy"
    #np.save(archivotiempos,np.array(ttotal/subject_samples))
    #print("Tiempo medio",np.mean(tiempos))
    print("Tiempo medio total",np.mean(ttotal/subject_samples))
    
    #print("Numero total desplazamientos",t)
    return X_all_rec
def generate_and_save_time_series_fromMTF(fold, dataset_folder, training_data, y_data, sj_train,dictionary, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):

    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))
    X_all_rec=[]
    tiempos=[]
    start_gl=time.time()
   
    #Colocar timer
    
    
    for i in p_bar:
      #start=time.time()
      w = training_data[i]
      sj = sj_train[i][0]
      w_y = y_data[i]
      w_y_no_cat = np.argmax(w_y)
      #print("w_y", w_y, "w_y_no_cat", w_y_no_cat)
      #print("w", w.shape)
      
      # Update Progress Bar after a while
      #time.sleep(0.01)
      #p_bar.set_description(f'[{data_type} | FOLD {fold} | Class {w_y_no_cat}] Subject {sj}')
      path=f"{dataset_folder}plots/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
      path=f"{path}{sj}x{subject_samples}mtf.png"  
      imagen = cv2.imread(path)  
      imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
      ##We need to change path 
      mtf=mtf2.Reconstruct_MTF(imagen,dictionary,subject_samples)
      #end=time.time()
      
      #Si queremos guardar los datos
      path=f"{dataset_folder}tseries/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/{sj}x{subject_samples}.npy"
      np.save(path, np.array(mtf))
      
      #print(rp)
      X_all_rec.append(mtf)
      #print("X_ALL",np.array(X_all_rec))
      #tiempo=end-start
      #tiempos.append(tiempo)
      #Guardar la rp en el path indicado con un nombre adecuado
      #print(tiempos)
      #w=w.reshape(3,129)
      #w=w[:, 0]
      #w=w[1:]
      #print("Forma del RP original y calculada",w.shape,rp.shape)
      
      #print("normal",w,"calculada",rp)
      
      error_abs,error_r,error_q,error_std,error_p,te=err.ts_error(w,mtf)
      
      
      #print("errores",np.array(error))
      #print("errores",np.array(errores))
      
      subject_samples += 1

    end_gl=time.time()
    ttotal=end_gl-start_gl

    #maybe we should calculate here the global of errors and the mean.
    #maybe we should calculate here the global of errors and the mean.
    archivoX_all=f"{dataset_folder}tseries/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/X_all_mtf.npy"
    np.save(archivoX_all,np.array(X_all_rec))
    
    #archivotiempos=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/tiempos_rec.npy"
    #np.save(archivotiempos,np.array(ttotal/subject_samples))
    #print("Tiempo medio",np.mean(tiempos))
    print("Tiempo medio total",np.mean(ttotal/subject_samples))
    
    #print("Numero total desplazamientos",t)
    return X_all_rec
#Generates all time series
def generate_all_time_series(X_train, y_train, sj_train, dataset_folder="/home/adriano/Escritorio/TFG/data/WISDM/", TIME_STEPS=129,  FOLDS_N=3, sampling="loso",reconstruction="all"):
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
        os.makedirs(f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}tseries/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}tseries/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}tseries/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}tseries/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}tseries/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
        os.makedirs(f"{dataset_folder}tseries/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)
  
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
    
    dictionary=dict()
    for k in range(0,training_data.shape[0]):
         l=training_data[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]
         
    print("training_data.shape", training_data.shape, "y_training_data.shape", y_training_data.shape, "sj_training_data.shape", sj_training_data.shape)
    print("validation_data.shape", validation_data.shape, "y_validation_data.shape", y_validation_data.shape, "sj_validation_data.shape", sj_validation_data.shape)
    

    ##aqui añadir una opcion si quieres todas o una en especifico
    if reconstruction=="MTF":
      generate_and_save_time_series_fromMTF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      
      #arch_training_data=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/training_data.npy"
      #np.save(arch_training_data,np.array(training_data))
      print("se hace 1 vez")
      break
    if reconstruction=="GAF":
      generate_and_save_time_series_fromGAF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      print("se hace 1 vez")
      break
    if reconstruction=="RP":
      generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)

      print("se hace 1 vez")
      break
    if reconstruction=="all":
      generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      generate_and_save_time_series_fromGAF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      generate_and_save_time_series_fromMTF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
  
      print("se hace 1 vez")
      break
     