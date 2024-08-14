#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2

import sklearn.metrics as metrics
import activity_data as act
import recurrence_plots as rec
import calculate_ts_errors as err
def RemoveZero(l):
    nonZeroL = []
    #nonZeroL = []
    for i in range(len(l)):
        if l[i] != 0.0:
            nonZeroL.append(l[i])
    return nonZeroL
#a = [0,-1,0.02,3]
#print RemoveZero(a)
def NormalizeMatrix(_r):
    dimR = _r.shape[0]
    #print(_r)
    h_max = []
    for i in range(dimR):
        h_max.append(max(_r[i]))
    _max =  max(h_max)
    h_min = []
    for i in range(dimR):
        #print _r[i]
        h_min.append(min(RemoveZero(_r[i])))
    
    _min =  min(h_min)
    _max_min = _max - _min
    _normalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _normalizedRP[i][j] = (_r[i][j]-_min)/_max_min
    return _normalizedRP





def main():
     
     data_name="WISDM"
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/"
     #voy a  obtener el maximo de todo el data set.
     X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
     print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
     #print(sj_train[:,0])
     #print(y_train[:,0])
     #print(X_train)
    
     MAX=np.max(X_train)
     MIN=np.min(X_train)
     #he obtenido el máximo y el minimo del dataset minimo -78.47761 maximo 66.615074
     a=0
     w = X_train[a]
     sj = sj_train[a][0]
     w_y = y_train[a]
     w_y_no_cat = np.argmax(w_y)
     dictionary=dict()
     for k in range(0,8163):
         l=X_train[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]
     #print("Diccionario",dictionary)
     
     img = rec.SavevarRP_XYZ(w, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     #parte de reconstruccion
     #primero genero la RP a partir de la imagen
     imagen = cv2.imread("./1600x0.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     rp,ninv=rec.Reconstruct_RP(imagen,dictionary,a)
     dim=1
     print(rp.shape)
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
     plt.legend(fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
     plt.clf()
     
     f=np.array(w[:,dim])
     f=f[1:]
     print(f.shape)
     error_absoluto, error_relativo = err.calcular_errores(f, rp[dim])
     d = metrics.mean_squared_error(f,rp[dim])#metrics.root_mean_squared_error()
     print(f"Error Absoluto Promedio: {error_absoluto}")
     print(f"Error Relativo Promedio: {error_relativo}")
     print(f"Error quadratico: {d}")
     print(f"Coeficiente de correlación: {np.corrcoef(f, rp[dim])[0,1]}")
    
     
# Guardar el gráfico como una imagen
    

# Mostrar el gráfico (opcional)
plt.show()
if __name__ == '__main__':
    main()