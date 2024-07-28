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
    data_name="WISDM"
    data_folder="/home/adriano/Escritorio/TFG/data/WISDM/"
     #voy a  obtener el maximo de todo el data set.
    X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
    
    data_folder="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
    data_F1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/recurrence_plot/"
    errores = np.load(f"{data_F1}/errores_rec2.npy")
    
    #data_F1="/home/adriano/Escritorio/TFG/data/WISDM/tseries/GAF/"
    #errores = np.load(f"{data_F1}/errores_gaf2.npy")
    #errores2 = np.load(f"{data_folder}/errores_rec.npy")
    indices_T=np.load(f"{data_folder}/training_data.npy")
    X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
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
    print("min r Promedio",(errores[:,2,:][errores[:,2,:]>450]))    
    indices=np.where(errores[:,2,:]>450) #PEOR PEARSON 2690 mejor pearson 2026
    indices = list(zip(indices[0], indices[1]))
    print(indices)
    print(errores[940,:,:])# 4913 candidata a mejor ideal ERROR relativo 
   
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
    w=indices_T[940]
    rp=X_all_rec[940]
     
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


     
    # Configurar el estilo de los gráficos
    plt.style.use("ggplot")  

    # Gráfico original
    plt.figure(figsize=(10, 6))
    plt.plot(w[1:, dim], marker='o', color='blue')
    plt.title('Original', fontsize=18,fontweight="bold")
    plt.xlabel("Tiempo", fontsize=12)
    plt.ylabel('Índice X', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('original.png', bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Gráfico reconstrucción
    plt.figure(figsize=(10, 6))
    plt.plot(rp[dim], marker='o', color='green')
    plt.title('Reconstrucción', fontsize=18,fontweight="bold")
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Índice X', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reconstruccion.png', bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Gráfico comparativa
    plt.figure(figsize=(10, 6))
    plt.plot(w[1:, dim], marker='o', label='Original', color='blue')
    plt.plot(rp[dim], marker='o', label='Reconstrucción', color='green')
    plt.title('Comparativa', fontsize=18,fontweight="bold")
    plt.xlabel("Tiempo", fontsize=12)
    plt.ylabel('Índice X', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
    
    f=np.array(w[:,dim])
    f=f[1:]
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
