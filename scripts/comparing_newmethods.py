#!/home/adriano/Escritorio/TFG/venv/bin/python3

from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
#import activity_data as act
import src.activity_data as act
#import recurrence_plots as rec
import src.recurrence_plots as rec
import cv2
import os
import time
import seaborn as sns
def main():
     data_name="WISDM"
     data_folder="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
     #voy a  obtener el maximo de todo el data set.
     X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
     print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
     #print(sj_train[:,0])
     #print(y_train[:,0])
     #print(X_train)
     print("minimo",np.min(X_train))
     print("maximo",np.max(X_train))
     MAX=np.max(X_train)
     MIN=np.min(X_train)
     #he obtenido el máximo y el minimo del dataset minimo -78.47761 maximo 66.615074
     a=0
     w = X_train[a]
     sj = sj_train[a][0]
     w_y = y_train[a]
     w_y_no_cat = np.argmax(w_y)
     
     print(w.shape)
     dictionary=dict()
     for k in range(0,8163):
         l=X_train[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]
     img = rec.SavevarRP_XYZ(w, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=128) 
     
     
     
     imagen = cv2.imread("./1600x0.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     rp,a=rec.Reconstruct_RP(imagen,dictionary,a)
     # Configurar el estilo de los gráficos
     print("rp",rp,a)
     dim=1
     
    # Configurar el estilo de los gráficos
     plt.style.use("ggplot")  

    # Gráfico original
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, dim], marker='o', color='blue')
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
     plt.plot(w[:, dim], marker='o', label='Original', color='blue')
     plt.plot(rp[dim], marker='o', label='Reconstrucción', color='green')
     plt.title('Comparativa', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     #plt.legend(loc="upper left",bbox_to_anchor=(1,1))
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
     plt.clf()

    
if __name__ == '__main__':
    main()