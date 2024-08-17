#!/home/adriano/Escritorio/TFG/venv/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt



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
     plt.plot(rp[dim],linestyle="--", marker='', label='Reconstrucción', color='darkgreen')
     plt.title('Comparativa', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.legend(fontsize=12,loc="upper left")
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
     plt.clf()
