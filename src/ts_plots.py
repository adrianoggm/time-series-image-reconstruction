#!/home/adriano/Escritorio/TFG/venv/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_time_series_g(original,reconstruccion):
     
     w=original
     rp=reconstruccion
    # Configurar el estilo de los gráficos
     plt.style.use("ggplot")  

    # Gráfico comparativa
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, 0], marker='', label='Originalx', color='darkred')
     plt.plot(rp[0],linestyle="--", marker='', label='Reconstrucciónx', color='indianred')
     plt.plot(w[:, 1], marker='', label='Originaly', color='darkgreen')
     plt.plot(rp[1],linestyle="--", marker='', label='Reconstruccióny', color='lime')
     plt.plot(w[:, 2], marker='', label='Originalz', color='darkblue')
     plt.plot(rp[2],linestyle="--", marker='', label='Reconstrucciónz', color='royalblue')
     plt.title('Comparativa', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.legend(fontsize=12,loc="upper left",bbox_to_anchor=(1,1))
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa_g.png', bbox_inches='tight', pad_inches=0)
     plt.clf()
def plot_time_series(original,reconstruccion,dim):
     dim=0
     w=original
     rp=reconstruccion
    # Configurar el estilo de los gráficos
     plt.style.use("ggplot")  

    # Gráfico original
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, dim], marker='', color='blue')
     plt.title('Original', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('original.png', bbox_inches='tight', pad_inches=0)
     plt.clf()

    # Gráfico reconstrucción
     plt.figure(figsize=(10, 6))
     plt.plot(rp[dim], marker='', color='darkgreen')
     plt.title('Reconstrucción', fontsize=18,fontweight="bold")
     plt.xlabel('Tiempo', fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('reconstruccion.png', bbox_inches='tight', pad_inches=0)
     plt.clf()

    # Gráfico comparativa
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, dim], marker='', label='Original', color='blue')
     plt.plot(rp[dim],linestyle="--", marker='', label='Reconstrucción', color='magenta')
     plt.title('Comparativa', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.legend(fontsize=12,loc="upper left")
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
     plt.clf()
def plot_cm(labels, pre):
    all_label_names=["Walking","Jogging","Stairs","Sitting","Standing"]
    conf_numpy = confusion_matrix(labels, pre)  # Calcular la matriz de confusión
    conf_df = pd.DataFrame(conf_numpy, index=all_label_names, columns=all_label_names)  # Crear un DataFrame
    
    plt.figure(figsize=(8,7))
    sns.heatmap(conf_df, annot=True, fmt="d", cmap="BuPu")  # Dibujar la matriz de confusión
    plt.title('Confusion Matrix', fontsize=15)
    plt.ylabel('True Value', fontsize=14)
    plt.xlabel('Predicted Value', fontsize=14)
    plt.savefig('ConfMatrix.png', bbox_inches='tight', pad_inches=0)
    plt.clf()