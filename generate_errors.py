#!/home/adriano/Escritorio/TFG/venv/bin/python3
from keras.utils import to_categorical
import argparse
import tgen.activity_data as act
import tgen.calculate_ts_errors as cerr
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import tgen.ts_plots as plot
from PIL import Image
from dtaidistance import dtw
import cv2
def main():
    data_name="WISDM"
    data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
     #voy a  obtener el maximo de todo el data set.
    img_type="MTF"
    indices_T=np.load(f"{data_folder}/training_data.npy")
    
    if img_type=="RP":
        data_F1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/"
        errores = np.load(f"{data_F1}/errores_rec.npy")
        X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
    if img_type=="GAF":
        data_F1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train"
        errores = np.load(f"{data_F1}/errores_gaf.npy")
        X_all_rec=np.load(f"/home/adriano/Escritorio/TFG/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train/X_all_gaf.npy")
        #errores2 = np.load(f"{data_folder}/errores_rec.npy")
    #data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
    if img_type=="MTF":
        data_F1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train"
        errores = np.load(f"{data_F1}/errores_mtf.npy")
        X_all_rec=np.load(f"/home/adriano/Escritorio/TFG/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train/X_all_mtf.npy")
    #print(sj_train[:,0])
     #print(f"Error Absoluto Promedio: {error_absoluto}")
      #print(f"Error Relativo Promedio: {error_relativo}")
      #print(f"Error DTW: {d}")
      #print(f"Coeficiente de correlación: {pearson}")
    tipoerrores=["Error Absoluto Promedio","Error Relativo Promedio","Error Quadrático medio","Error desviacion típica","Coeficiente de correlación pearson"]
    print("errores",errores.shape)
    for i in range(0,len(tipoerrores)):
        #if(j!=1):
            a=tipoerrores[i]
            for j in range(0,3):
                
                print("ERROR TIPO ",a,"en dim",j,"es :",np.mean(errores[:,i,j][errores[:,i,j]<=1000]))
                print ("Desviación típica de ERROR TIPO",a,"en dim",j,"es :",np.std(errores[:,i,j][errores[:,i,j]<=1000]))
            print("ERROR TIPO ",a," medio global es :",np.mean(errores[:,i,:][errores[:,i,:]<=1000]))
            print ("Desviación típica de ERROR TIPO",a,"es :",np.std(errores[:,i,:][errores[:,i,:]<=1000]))
            """else :
                for j in range(0,3):
                    
                    print("ERROR TIPO {tipoerrores[i]}en dim {j}es :",np.mean(errores[:,i,j]))
                print("ERROR TIPO {tipoerrores[i]} medio global es :",np.mean(errores[:,i,:]))
            """
    print("min r Promedio",(errores[:,4,:][errores[:,4,:]<4.09187097e-05]),np.min(errores[:,4,:]))    
    indices=np.where(errores[:,4,:]<4.09187097e-05) #PEOR PEARSON 2690 mejor pearson 2026
    indices = list(zip(indices[0], indices[1]))
    print(indices)
    print(errores[3301,:,:])# 4913 candidata a mejor ideal ERROR relativo 
   
    """
    mejor pearson 2012 dim 0
    mejor quadrático medio 4222, 1
    mejor Error desviacion típica 4227 dim 1 0.00498124


    peor    Error desviacion típica   (3656, 2) 19.13756184
    peor    Error pearson   69, 0 -0.05487599
    peor    error quadrático 3660,1

    PEOR   ERROR relativo 3068, 1  450.10440276
          (2380, 1
    """
    w=indices_T[3301]
    rp=X_all_rec[3301]
     
    dim=1
    """
     
     valoresa=np.linspace(-5, 20, 129)
     valoresb=np.linspace(-20, 50, 129)
     valoresc=np.linspace(-10, 3, 129)
     experimento=np.array([valoresa,valoresb,valoresc]).reshape(129,3)
     experimentoinv=np.array([valoresa[::-1],valoresb[::-1],valoresc[::-1]]).reshape(129,3)
     
     img = rec.SavevarRP_XYZ(experimento, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
     img2 = rec.SavevarRP_XYZ(experimentoinv, sj, 1, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     
     
        
        
     
     rp[0]=-1*rp[0]
     _max=np.max(w[:,0])
     _min=np.min(w[:,0])
     s=np.interp(rp[0],(np.min(rp[0]),np.max(rp[0])),(_min,_max)).reshape(128)
     _max=np.max(w[:,1])
     _min=np.min(w[:,1])
     s1=np.interp(rp[1],(np.min(rp[1]),np.max(rp[1])),(_min,_max)).reshape(128)
     _max=np.max(w[:,2])
     _min=np.min(w[:,2])
     s2=np.interp(rp[2],(np.min(rp[2]),np.max(rp[2])),(_min,_max)).reshape(128)
    """


     
    plot.plot_time_series(w,rp,dim)
    
    f=np.array(w[:,dim])
    f=f[:]
    print(f.shape)
    error_absoluto, error_relativo = cerr.calcular_errores(f, rp[dim])
    #d = dtw.distance_fast(f, rp[1], use_pruning=True)
    print(f"Error Absoluto Promedio: {error_absoluto}")
    print(f"Error Relativo Promedio: {error_relativo}")
    d = metrics.mean_squared_error(f,rp[dim]) 
    print(f"Error DTW: {d}")
    print(f"Coeficiente de correlación: {np.corrcoef(f, rp[dim])[0,1]}")
    
     
# Guardar el gráfico como una imagen
if __name__ == '__main__':
    main()
