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
import statistics
def main():
    data_name="WISDM"
    data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
     #voy a  obtener el maximo de todo el data set.
    img_type="RP"
    indices_T=np.load(f"{data_folder}/training_data.npy")
    
    if img_type=="RP":
        data_F1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
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
            
    print("min r Promedio",(errores[:,4,:][errores[:,4,:]==0.9999117348854251]),np.min(errores[:,4,:]))    
    indices=np.where(errores[:,4,:]==0.9999117348854251) #PEOR PEARSON 2690 mejor pearson 2026
    indices = list(zip(indices[0], indices[1]))
    print(indices)
    print(errores[3301,:,:])# 4913 candidata a mejor ideal ERROR relativo 
    
    pear=[]
    quad=[]

    for i in range(0,len(errores)):
        quad=np.append(quad,np.mean(errores[i,2,:]))
        pear=np.append(pear,np.mean(errores[i,4,:])) 
        
    
    media=statistics.mean(quad)
    d1=np.abs(quad-media)
    print("MEDIA Q",media,quad[np.argsort(d1)][0],np.argsort(d1)[0])#0
    media=statistics.mean(pear)
    d1=np.abs(pear-media)
    print("MEDIA P",media,pear[np.argsort(d1)][0],np.argsort(d1)[0])#0 resulta ser la 0 en ambas
    
    min=np.min(quad)
    d1=np.abs(quad-min)
    print("Min Q",min,quad[np.argsort(d1)][0],np.argsort(d1)[0])#1490
    min=np.min(pear)
    d1=np.abs(pear-min)
    print("Min P",min,pear[np.argsort(d1)][0],np.argsort(d1)[0])#0
    
    max=np.max(quad)
    d1=np.abs(quad-max)
    print("Max Q",max,quad[np.argsort(d1)][0],np.argsort(d1)[0])#0PEOR Q 2.51544503e-01 3661 
    max=np.max(pear)
    d1=np.abs(pear-max)
    print("Max P",max,pear[np.argsort(d1)][0],np.argsort(d1)[0])#00.99999221 2156
    """
    GAF
    MEDIA Q 0.0044012627779020045 0.004399589350389923 4170
    MEDIA P 0.9998733035276017 0.9998733283554287 5406
    Min Q 7.442646919590436e-09 7.442646919590436e-09 4225
    Min P 0.9912638600577192 0.9912638600577192 1490
    Max Q 0.25154450250116284 0.25154450250116284 3661
    Max P 0.9999922113685015 0.9999922113685015 2156


    MTF
    MEDIA Q 110.39107219960607 110.38338496650113 4221
    MEDIA P 0.547349329114101 0.5474256616048051 3873
    Min Q 0.00039825152344224795 0.00039825152344224795 4218
    Min P 0.03311714899700962 0.03311714899700962 1558
    Max Q 1780.8914552240483 1780.8914552240483 3662
    Max P 0.9450789819158428 0.9450789819158428 377

    RP
    MEDIA Q 15.936688647211787 15.94236676824787 2084
    MEDIA P 0.7731528141371725 0.7731454984152836 4085
    Min Q 0.00015950539571661464 0.00015950539571661464 4424
    Min P 0.05182189333660742 0.05182189333660742 1104
    Max Q 416.0807310831489 416.0807310831489 3660
    Max P 0.9588916226063297 0.9588916226063297 2156
    """
    w=indices_T[3301]
    rp=X_all_rec[3301]
     
    dim=1
  

     
    plot.plot_time_series(w,rp,dim)
    
   
     
# Guardar el gráfico como una imagen
if __name__ == '__main__':
    main()
