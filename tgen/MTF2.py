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
    mtf = MarkovTransitionField(image_size=num_states,overlapping=True,n_bins=num_states)
    
    X_mtf = mtf.fit_transform(x).reshape(num_states, num_states)
    
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
    #print(X.shape)
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
     _r = varMTF2(x,'x', TIME_STEPS)
     _g = varMTF2(x,'y', TIME_STEPS)
     _b = varMTF2(x,'z', TIME_STEPS)

     print("X", _r[0])
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
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
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
def Reconstruct_MTF(img):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    
    _r=np.interp(_r,(0,255),(0,1))
    _g=np.interp(_g,(0,255),(0,1))
    _b=np.interp(_b,(0,255),(0,1))
    print(_r)
    r=MTF_to_TS(_r,16,0)
    g=MTF_to_TS(_g,3,1)
    b=MTF_to_TS(_b,3,2)
    N=[]
    N.append(r)
    N.append(g)
    N.append(b)
    return N
def MTF_to_TS(mtf,initialstate=3,index=0,numstates=32,TIMESTEPS=129):
    mtf2=np.zeros_like(mtf)
    for i in range(0,numstates): 
        
        v=1-np.sum(mtf[i])
        if(v>=0):
            mtf2[i]=mtf[i]+(v/numstates)
            
        if(v<0):
            mtf2[i]=mtf[i]
            for j in range(0,numstates):
                if mtf2[i][j]+v>=0:
                    mtf2[i][j]+=v
                    break        
        print(np.sum(mtf2[i]))            
    
    states=np.arange(numstates)
    generated_series=[]
    generated_series.append(initialstate)
    current_state=initialstate
    np.random.seed(0)   
    for _ in range(0,TIMESTEPS-1):
        current_index = current_state
        next_state = np.random.choice(states, p=mtf2[current_index])
        generated_series.append(next_state)
        current_state = next_state
    
    #RECONSTRUCCIÓN DISCRETA DE LOS ESTADOS:
    #Sabemos que el valor medio de los datos maximos es 
    MAX=[10.657428709640488,3.0590269720681738,7.629156537079175] 
    MIN=[-4.128293834805366,-11.814377181580914,-5.316818145702738]

    tam=MAX[index]-MIN[index]
    estadosdiscretos=np.zeros(numstates)
    estadosdiscretos[0]=MIN[index]
    a=MIN[index]
    for i in range(1,numstates):
       a=a+tam/numstates
       estadosdiscretos[i]=a
    serie=np.zeros(128)
    for i in range(0,TIMESTEPS-1):
        serie[i]=estadosdiscretos[generated_series[i]]
    return serie
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
     imagen = cv2.imread("./1600x0mtf.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     rp=Reconstruct_MTF(imagen)
     # Configurar el estilo de los gráficos
     plt.style.use("ggplot")  

    # Gráfico original
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, 0], marker='o', color='blue')
     plt.title('Original', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('original.png', bbox_inches='tight', pad_inches=0)
     plt.clf()

    # Gráfico reconstrucción
     plt.figure(figsize=(10, 6))
     plt.plot(rp[0], marker='o', color='green')
     plt.title('Reconstrucción', fontsize=18,fontweight="bold")
     plt.xlabel('Tiempo', fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('reconstruccion.png', bbox_inches='tight', pad_inches=0)
     plt.clf()

    # Gráfico comparativa
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, 0], marker='o', label='Original', color='blue')
     plt.plot(rp[0], marker='o', label='Reconstrucción', color='green')
     plt.title('Comparativa', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.legend(fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
     plt.clf()
     
     f=np.array(w[:,0])
     f=f[1:]
     print(f.shape)
     error_absoluto, error_relativo = calcular_errores(f, rp[0])
     print(f"Error Absoluto Promedio: {error_absoluto}")
     print(f"Error Relativo Promedio: {error_relativo}")
     print(f"Coeficiente de correlación: {np.corrcoef(f, rp[0])[0,1]}")
plt.show()
if __name__ == '__main__':
    main()