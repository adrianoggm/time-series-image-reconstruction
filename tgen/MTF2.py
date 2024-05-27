#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import activity_data as act
import recurrence_plots as rec
from pyts.image import MarkovTransitionField
from scipy.sparse.csgraph import dijkstra
from PIL import Image
import cv2
from sklearn.manifold import MDS
def varMTF2(data, dim,TIME_STEPS):
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z':
        k=2
    
    x=np.array(data[:,k]).reshape(1,-1)
    num_states = TIME_STEPS//4
    mtf = MarkovTransitionField(image_size=TIME_STEPS,overlapping=True,n_bins=5)
    
    X_mtf = mtf.fit_transform(x).reshape(TIME_STEPS, TIME_STEPS)
    
    return X_mtf
def NormalizeMatrix_Adri(_r):
    dimR = _r.shape[0]
    _max=66.615074
    _min =  -78.47761
    _max_min = _max - _min
    #_normalizedRP=np.interp(_r,(_min,_max),(0,1))
    
    _normalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _normalizedRP[i][j] = (_r[i][j]-_min)/_max_min
    
    return _normalizedRP
def RGBfromMTFMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    print(X.shape)
    dimImage = X.shape[0]
    newImage = np.zeros((dimImage,dimImage,3))
    for i in range(dimImage):
        for j in range(dimImage):
            _pixel = []
            _pixel.append(X[i][j])
            _pixel.append(Y[i][j])
            _pixel.append(Z[i][j])
            newImage[i][j] = _pixel
    return newImage
def varMTF(data, dim,TIME_STEPS):
    
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z':
        k=2
    
    x=data[:,k]
    num_states = 5
    quantiles = np.quantile(x, [i/num_states for i in range(1, num_states)])
    discretized_series = np.digitize(x, quantiles)
    # Inicialización de la matriz de transición
    transition_matrix = np.zeros((num_states, num_states))
    #print(discretized_series.shape,discretized_series[:-1])
# Contar las transiciones entre estados
    for (i, j) in zip(discretized_series[:-1], discretized_series[1:]):
        transition_matrix[i, j] += 1
    
# Normalizar para convertir en probabilidades
    transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)
    #print(transition_matrix,transition_matrix.shape)
    n = len(discretized_series)

# Inicialización del MTF
    MTF = np.zeros((n, n))

# Llenar el MTF con las probabilidades de transición
    for i in range(n):
        for j in range(n):
            MTF[i, j] = transition_matrix[discretized_series[i], discretized_series[j]]
    print(MTF)
    return MTF 


def SavevarMTF_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     _r = varMTF(x,'x', TIME_STEPS)
     _g = varMTF(x,'y', TIME_STEPS)
     _b = varMTF(x,'z', TIME_STEPS)

     print("X", _r[0][0])
     #print("Y", _g[1][4])
     #print("Z", _b[1][4])
     #print("Y", _g)
     #print("Z", _b)
     
     # plt.close('all')
     # plt.figure(figsize=(1,1))
     # plt.axis('off')
     # plt.margins(0,0)
     # plt.gca().xaxis.set_major_locator(plt.NullLocator())
     # plt.gca().yaxis.set_major_locator(plt.NullLocator())

     #print("fig size: width=", plt.figure().get_figwidth(), "height=", plt.figure().get_figheight())

     if normalized:
          newImage = RGBfromMTFMatrix_of_XYZ(_r, _g, _b)
          #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          # print(newImage.shape)
          #print(newImage[1][4][0]* 255)
          newImage = Image.fromarray((newImage * 255).astype(np.uint8))
          # plt.imshow(newImage)
          
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
               newImage.save(f"{path}{sj}{action}{item_idx}mtf.png")
          # plt.close('all')
     else:
          newImage = RGBfromMTFMatrix_of_XYZ(_r, _g, _b)
          newImage = Image.fromarray((newImage * 255).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}mtf.png")
          # plt.close('all')
     return newImage
    else:
     return None
   
def calcular_errores(valores_verdaderos, valores_aproximados):
    # Convertir las listas a arrays de numpy para facilitar los cálculos
    valores_verdaderos = np.array(valores_verdaderos)
    valores_aproximados = np.array(valores_aproximados)
    
    # Calcular el error absoluto
    errores_absolutos = np.abs(valores_verdaderos - valores_aproximados)
    
    # Calcular el error relativo (evitando la división por cero)
    errores_relativos = np.abs(errores_absolutos / valores_verdaderos)
    
    # Calcular el error absoluto promedio
    error_absoluto_promedio = np.mean(errores_absolutos)
    
    # Calcular el error relativo promedio
    error_relativo_promedio = np.mean(errores_relativos)
    
    return error_absoluto_promedio, error_relativo_promedio

def main():
     data_name="WISDM"
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/"
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
     w = X_train[3]
     sj = sj_train[0][0]
     w_y = y_train[1]
     w_y_no_cat = np.argmax(w_y)
     print(w.shape)
     img = SavevarMTF_XYZ(w, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
     
plt.show()
if __name__ == '__main__':
    main()