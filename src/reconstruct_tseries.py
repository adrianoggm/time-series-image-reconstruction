
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
import src.calculate_ts_errors as err
import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import src.recurrence_plots as rpts
import src.GAF as gaf
import src.GADF as gadf
#import tgen.GAF2 as gaf2
import time

from src.config import DEFAULT_DATA_DIR
import src.MTF as mtf2
import os    

""" Generates and saves a time series from a RP saving keeping also 
    track of the acuracy and the metrics from the original Time series
"""
"""""
def generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_data, sj_train,dictionary, TIME_STEPS=128, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
    
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
"""

from concurrent.futures import ProcessPoolExecutor, as_completed

def _process_sample(i, fold, dataset_folder, dictionary, training_data, y_data, sj_train, sampling, FOLDS_N, data_type):
    """Reconstruct one window's RP image and persist it under tseries/.

    Returns ``(i, rp, a)`` — the index is carried back explicitly so the
    caller can re-sort the parallel results, since ``as_completed`` does NOT
    preserve submission order.
    """
    sj = sj_train[i][0]
    w_y = y_data[i]
    w_y_no_cat = np.argmax(w_y)

    path_img = (
        f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/"
        f"{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
        f"{sj}x{i}.png"
    )
    imagen = cv2.imread(path_img)
    if imagen is None:
        raise FileNotFoundError(f"No se encontró la imagen: {path_img}")
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    rp, a = rpts.Reconstruct_RP(imagen, dictionary, i)

    path_npy = (
        f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/"
        f"{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
        f"{sj}x{i}.npy"
    )
    np.save(path_npy, np.array(rp))

    return i, rp, a

def generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_data, sj_train,
                                           dictionary, TIME_STEPS=128, data_type="train",
                                           single_axis=False, FOLDS_N=3, sampling="loso"):
    """
    Genera y guarda series temporales a partir de RP, calculando además métricas como el
    número total de inversiones. Se procesa cada muestra en paralelo.
    """
    errores = []  # Se mantiene la variable para compatibilidad
    ninv = 0
    start_gl = time.time()
    n_samples = len(training_data)

    # Process windows in parallel; collect (i, rp) so we can re-sort at the
    # end (as_completed does NOT preserve submission order, and the aggregate
    # X_all_rec.npy must align with y_validation_data row-by-row).
    indexed_results: list[tuple[int, object]] = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_process_sample, i, fold, dataset_folder, dictionary,
                            training_data, y_data, sj_train, sampling, FOLDS_N, data_type)
            for i in range(n_samples)
        ]
        for future in tqdm(as_completed(futures), total=n_samples, desc="Procesando muestras"):
            i, rp, a = future.result()
            indexed_results.append((i, rp))
            ninv += a

    # Restore submission order before stacking.
    indexed_results.sort(key=lambda p: p[0])
    X_all_rec = [rp for _i, rp in indexed_results]

    end_gl = time.time()
    ttotal = end_gl - start_gl

    archivoX_all = (
        f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/"
        f"{FOLDS_N}-fold/fold-{fold}/{data_type}/X_all_rec.npy"
    )
    np.save(archivoX_all, np.array(X_all_rec))

    print("Tiempo medio total por muestra:", ttotal / n_samples)
    print("Número total de inversiones:", ninv)

    return X_all_rec, errores

def generate_and_save_time_series_fromGAF(fold, dataset_folder, training_data, y_data, sj_train,dictionary, TIME_STEPS=128, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):

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
def generate_and_save_time_series_fromGADF(fold, dataset_folder, training_data, y_data, sj_train, dictionary, TIME_STEPS=128, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))
    X_all_rec = []
    start_gl = time.time()

    for i in p_bar:
        w_y = y_data[i]
        w_y_no_cat = np.argmax(w_y)
        sj = sj_train[i][0]

        path = f"{dataset_folder}plots/GADF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
        path = f"{path}{sj}x{subject_samples}gadf.png"
        imagen = cv2.imread(path)
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        gadf_ts = gadf.Reconstruct_GADF(imagen, dictionary, subject_samples)

        out_path = f"{dataset_folder}tseries/GADF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/{sj}x{subject_samples}.npy"
        np.save(out_path, np.array(gadf_ts))

        X_all_rec.append(gadf_ts)
        subject_samples += 1

    ttotal = time.time() - start_gl
    archivoX_all = f"{dataset_folder}tseries/GADF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/X_all_gadf.npy"
    np.save(archivoX_all, np.array(X_all_rec))
    print("Tiempo medio total", np.mean(ttotal / max(subject_samples, 1)))
    return X_all_rec

def _process_sample_mtf(i, fold, dataset_folder, dictionary, y_data, sj_train, sampling, FOLDS_N, data_type):
    """Per-window MTF reconstruction worker (parallel-safe).

    Returns ``(i, mtf)`` — the index is carried back so the caller can
    re-sort the parallel results (``as_completed`` does NOT preserve order).
    """
    sj = sj_train[i][0]
    w_y_no_cat = int(np.argmax(y_data[i]))

    path_img = (
        f"{dataset_folder}plots/MTF/sampling_{sampling}/"
        f"{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
        f"{sj}x{i}mtf.png"
    )
    imagen = cv2.imread(path_img)
    if imagen is None:
        raise FileNotFoundError(f"missing image: {path_img}")
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    mtf = mtf2.Reconstruct_MTF(imagen, dictionary, i)

    path_npy = (
        f"{dataset_folder}tseries/MTF/sampling_{sampling}/"
        f"{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/"
        f"{sj}x{i}.npy"
    )
    np.save(path_npy, np.array(mtf))
    return i, mtf


def generate_and_save_time_series_fromMTF(fold, dataset_folder, training_data, y_data, sj_train,
                                          dictionary, TIME_STEPS=128, data_type="train",
                                          single_axis=False, FOLDS_N=3, sampling="loso"):
    """MTF reconstruction over all windows, parallelised across CPU cores.

    Mirrors the RP refactor: each window is processed in a worker, results
    keep their index so the aggregate `X_all_mtf.npy` stays aligned with
    `y_validation_data.npy`.
    """
    start_gl = time.time()
    n_samples = len(training_data)
    indexed_results: list[tuple[int, object]] = []

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_process_sample_mtf, i, fold, dataset_folder, dictionary,
                            y_data, sj_train, sampling, FOLDS_N, data_type)
            for i in range(n_samples)
        ]
        for future in tqdm(as_completed(futures), total=n_samples, desc="MTF reconstruction"):
            i, mtf = future.result()
            indexed_results.append((i, mtf))

    indexed_results.sort(key=lambda p: p[0])
    X_all_rec = [mtf for _i, mtf in indexed_results]

    ttotal = time.time() - start_gl

    #maybe we should calculate here the global of errors and the mean.
    #maybe we should calculate here the global of errors and the mean.
    archivoX_all=f"{dataset_folder}tseries/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/X_all_mtf.npy"
    np.save(archivoX_all,np.array(X_all_rec))

    if n_samples:
        print("Tiempo medio por ventana:", ttotal / n_samples)
    return X_all_rec
#Generates all time series
def generate_all_time_series(X_train, y_train, sj_train, dataset_folder=DEFAULT_DATA_DIR, TIME_STEPS=128,  FOLDS_N=3, sampling="loso",reconstruction="all", target_fold=None):
  # When ``target_fold`` is set, every fold whose index does not match is
  # skipped. The historical behaviour (run fold-0 only) is preserved by the
  # ``break`` statements below; passing ``target_fold=1`` or ``2`` lets the
  # orchestrator drive each remaining fold from outside.
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
        os.makedirs(f"{dataset_folder}tseries/GADF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
        os.makedirs(f"{dataset_folder}tseries/GADF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)
        os.makedirs(f"{dataset_folder}tseries/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
        os.makedirs(f"{dataset_folder}tseries/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)
  
  for fold, (train_index, val_index) in enumerate(sgkf.split(X_train, y_train_no_cat, groups=groups)):

    if target_fold is not None and fold != target_fold:
      continue

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
    # Generar diccionario para validación
    dictionary_val = dict()
    for k in range(validation_data.shape[0]):
        serie = validation_data[k]
        maximos = [np.max(serie[:,0]), np.max(serie[:,1]), np.max(serie[:,2])]
        minimos = [np.min(serie[:,0]), np.min(serie[:,1]), np.min(serie[:,2])]
        dictionary_val[k] = [maximos, minimos]     

    print("training_data.shape", training_data.shape, "y_training_data.shape", y_training_data.shape, "sj_training_data.shape", sj_training_data.shape)
    print("validation_data.shape", validation_data.shape, "y_validation_data.shape", y_validation_data.shape, "sj_validation_data.shape", sj_validation_data.shape)
    arch_training_data=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/training_data.npy"
    np.save(arch_training_data,np.array(training_data))
    arch_training_data=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/y_training_data.npy"
    np.save(arch_training_data,np.array( y_training_data))
        
        # Guardar splits de entrenamiento
    arch_training_data = f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/training_data.npy"
    np.save(arch_training_data, np.array(training_data))
    arch_y_training_data = f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/y_training_data.npy"
    np.save(arch_y_training_data, np.array(y_training_data))
    arch_sj_training_data = f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/sj_training_data.npy"
    np.save(arch_sj_training_data, np.array(sj_training_data))

    # Guardar splits de validación
    arch_validation_data = f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/validation_data.npy"
    np.save(arch_validation_data, np.array(validation_data))
    arch_y_validation_data = f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/y_validation_data.npy"
    np.save(arch_y_validation_data, np.array(y_validation_data))
    arch_sj_validation_data = f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/sj_validation_data.npy"
    np.save(arch_sj_validation_data, np.array(sj_validation_data))

    ##aqui añadir una opcion si quieres todas o una en especifico
    if reconstruction=="MTF":
      #generate_and_save_time_series_fromMTF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      generate_and_save_time_series_fromMTF(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data,dictionary_val,TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      print("se hace 1 vez")
      break
    if reconstruction=="GAF":

      #generate_and_save_time_series_fromGAF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      generate_and_save_time_series_fromGAF(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data,dictionary_val,TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      print("se hace 1 vez")
      break
    if reconstruction=="GADF":
      generate_and_save_time_series_fromGADF(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, dictionary_val, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      print("se hace 1 vez")
      break
    if reconstruction=="RP":
      generate_and_save_time_series_fromRP(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, dictionary_val, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      print("se hace 1 vez")
      break
    if reconstruction=="all":
      generate_and_save_time_series_fromRP(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      generate_and_save_time_series_fromGAF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
      generate_and_save_time_series_fromMTF(fold, dataset_folder, training_data, y_training_data, sj_training_data,dictionary,TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
  
      print("se hace 1 vez")
      break
     