#!/home/adriano/Escritorio/TFG/venv/bin/python3
from keras.utils import to_categorical
import argparse
import tgen.activity_data as act
import tgen.calculate_ts_errors as cerr
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dtaidistance import dtw
import cv2
def main():
    

    data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
    data_folder1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train"
    data_folder2="/home/adriano/Escritorio/TFG/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train"
   
    errores_rec = []
    errores_gasf = []
    errores_mtf = []
    
    indices_T=np.load(f"{data_folder}/training_data.npy")
    X_all_gaf=np.load(f"{data_folder1}/X_all_gaf.npy")
    X_all_mtf=np.load(f"{data_folder2}/X_all_mtf.npy")
    X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
    
    
    t=0
    t1=0
    t2=0
    for i in range(0,len(indices_T[:,0,0])):
         w=indices_T[i]
         rp=X_all_mtf[i]
         error=[]
         error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp,flag=False)
         t+=te
         error.append(error_abs)
         error.append(error_r)
         error.append(error_q)
         error.append(error_std)
         error.append(error_p)
         errores_mtf.append(error)
         
         rp=X_all_gaf[i]
         error=[]
         error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp)
         t1+=te
         error.append(error_abs)
         error.append(error_r)
         error.append(error_q)
         error.append(error_std)
         error.append(error_p)
         errores_gasf.append(error)
         
         rp=X_all_rec[i]
         error=[]
         error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp)
         t2+=te
         error.append(error_abs)
         error.append(error_r)
         error.append(error_q)
         error.append(error_std)
         error.append(error_p)
         errores_rec.append(error)


    print("Desplazamientos en el caso de t son inversiones :",t,t1,t2)
    archivoerrores=f"{data_folder2}/errores_mtf.npy"
    np.save(archivoerrores,np.array(errores_mtf)) 
    archivoerrores=f"{data_folder1}/errores_gaf.npy"    
    np.save(archivoerrores,np.array(errores_gasf))
    archivoerrores=f"{data_folder}/errores_rec.npy"
    np.save(archivoerrores,np.array(errores_rec))
    
if __name__ == '__main__':
        main()