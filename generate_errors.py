#!/home/adriano/Escritorio/TFG/venv/bin/python3
from keras.utils import to_categorical
import argparse
import tgen.activity_data as act
import tgen.calculate_ts_errors as cerr
import tgen.REC as rec
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
    errores = np.load(f"{data_folder}/errores_rec.npy")
    indices_T=np.load(f"{data_folder}/training_data.npy")
    X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
    #print(sj_train[:,0])
     #print(f"Error Absoluto Promedio: {error_absoluto}")
      #print(f"Error Relativo Promedio: {error_relativo}")
      #print(f"Error DTW: {d}")
      #print(f"Coeficiente de correlación: {pearson}")
    tipoerrores=["Error Absoluto Promedio","Error Relativo Promedio","Error DTW","Coeficiente de correlación pearson"]
    print("errores",errores.shape)
    for i in range(0,4):
        #if(j!=1):
            a=tipoerrores[i]
            for j in range(0,3):
                
                print("ERROR TIPO ",a,"en dim",j,"es :",np.mean(errores[:,i,j][errores[:,i,j]<=1000]))
            print("ERROR TIPO ",a," medio global es :",np.mean(errores[:,i,:][errores[:,i,:]<=1000]))
            """else :
                for j in range(0,3):
                    
                    print("ERROR TIPO {tipoerrores[i]}en dim {j}es :",np.mean(errores[:,i,j]))
                print("ERROR TIPO {tipoerrores[i]} medio global es :",np.mean(errores[:,i,:]))
            """
    print("min r Promedio",(errores[:,3,:][errores[:,3,:]>0.998]))    
    indices=np.where(errores[:,3,:]>0.998) #PEOR PEARSON 2690 mejor pearson 2026
    indices = list(zip(indices[0], indices[1]))
    print(indices)
    print(errores[2026,:,:])# 4913 candidata a mejor ideal ERROR relativo 
    """
    peor pearson
        Error Absoluto Promedio: 5.469057299217118
        Error Relativo Promedio: 4.24974313362862
        Error DTW: 61.94575499104359
        Coeficiente de correlación: -0.842975121758349 

        [[ 14.35253996   1.68042245   1.08356166]
 [  3.24866685   0.6100595    1.7951847 ]
 [156.73305027  11.53423453   8.3628941 ]
 [  0.99842605   0.99725493   0.99747166]]



    """
    w=indices_T[2690]
    rp=X_all_rec[2690]
     
    # Configurar el estilo de los gráficos
    plt.style.use("ggplot")  

    # Gráfico original
    plt.figure(figsize=(10, 6))
    plt.plot(w[:, 1], marker='o', color='blue')
    plt.title('Original', fontsize=18,fontweight="bold")
    plt.xlabel("Tiempo", fontsize=12)
    plt.ylabel('Índice X', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('original.png', bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Gráfico reconstrucción
    plt.figure(figsize=(10, 6))
    plt.plot(rp[1], marker='o', color='green')
    plt.title('Reconstrucción', fontsize=18,fontweight="bold")
    plt.xlabel('Tiempo', fontsize=12)
    plt.ylabel('Índice X', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reconstruccion.png', bbox_inches='tight', pad_inches=0)
    plt.clf()

    # Gráfico comparativa
    plt.figure(figsize=(10, 6))
    plt.plot(w[:, 1], marker='o', label='Original', color='blue')
    plt.plot(rp[1], marker='o', label='Reconstrucción', color='green')
    plt.title('Comparativa', fontsize=18,fontweight="bold")
    plt.xlabel("Tiempo", fontsize=12)
    plt.ylabel('Índice X', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
    plt.clf()
     
    f=np.array(w[:,1])
    f=f[:-1]
    print(f.shape)
    error_absoluto, error_relativo = rec.calcular_errores(f, rp[1])
    d = dtw.distance_fast(f, rp[1], use_pruning=True)
    print(f"Error Absoluto Promedio: {error_absoluto}")
    print(f"Error Relativo Promedio: {error_relativo}")
    print(f"Error DTW: {d}")
    print(f"Coeficiente de correlación: {np.corrcoef(f, rp[1])[0,1]}")
     
     
# Guardar el gráfico como una imagen
if __name__ == '__main__':
    main()
