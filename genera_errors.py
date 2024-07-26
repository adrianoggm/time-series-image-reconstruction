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
    errores = []
    indices_T=np.load(f"{data_folder}/training_data.npy")
    X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
    
    
    t=0
    for i in range(0,len(indices_T[:,0,0])):
         w=indices_T[i]
         rp=X_all_rec[i]
         error=[]
         error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp)
         t+=te
         error.append(error_abs)
         error.append(error_r)
         error.append(error_q)
         error.append(error_std)
         error.append(error_p)
         errores.append(error)
    print("Desplazamientos :",t)
    dataset_folder="/home/adriano/Escritorio/TFG/data/WISDM/"     
    archivoerrores=f"{dataset_folder}tseries/recurrence_plot/errores_rec2.npy"
    np.save(archivoerrores,np.array(errores))
    
if __name__ == '__main__':
        main()