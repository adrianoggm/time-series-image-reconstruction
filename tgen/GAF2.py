#!/home/adriano/Escritorio/TFG/venv/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
import activity_data as act
import calculate_ts_errors as err
from pyts.image import GramianAngularField
from scipy.sparse.csgraph import dijkstra
from PIL import Image
import cv2
from sklearn.manifold import MDS
from scipy.optimize import fsolve

def normalize_to_zero_one(series):
    """
    Normaliza una serie temporal al rango [0, 1].
    
    Args:
    series (np.array): Serie temporal a normalizar.
    
    Returns:
    np.array: Serie temporal normalizada al rango [0, 1].
    """
    min_val = np.min(series)
    max_val = np.max(series)
    normalized_series = (series - min_val) / (max_val - min_val)
    normalized_series = np.where(normalized_series >= 1., 1., normalized_series)
    normalized_series = np.where(normalized_series <= 0., 0., normalized_series)
    return normalized_series
def tabulate(x, y, f):
    """Return a table of f(x, y). Useful for the Gram-like operations."""
    return np.vectorize(f)(*np.meshgrid(x, y, sparse=True))

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))
def sin_diff(a, b):
    """To work with tabulate."""
    return(math.sin(a-b))
def normalize_series(series):
    min_val = np.min(series)
    max_val = np.max(series)
    # Floating point inaccuracy!
    scaled_serie =2 * (series - min_val) / (max_val - min_val) - 1
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)
    return scaled_serie

def gramian_angular_field(series, method='summation'):
    # Normalización de la serie temporal
    normalized_series = normalize_series(series)
    #print("SERIE_normalizada_R",normalized_series)
    # Conversión a ángulos
    # Polar encoding
    #print("Serie norm",series,normalized_series)
    phi = np.arccos(normalized_series) 
    #print("Serie norm",phi,normalized_series)
    #print("Angulos",phi)
    #print("valores de PHI de R",phi)
    """
    Para la reconstruccion nos servirá
    r = np.linspace(0, 1, len(normalized_series))
    print("Radius",r)
    """
    # Construcción del GAF
    if method == 'summation':
        gaf = tabulate(phi, phi, cos_sum)
    elif method == 'difference':
        gaf = tabulate(phi, phi, sin_diff)
    else:
        raise ValueError("Method must be 'summation' or 'difference'")
    
    return gaf
    

def varGAF(data, dim,TIME_STEPS):
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z': 
        k=2
    
    
    f=data[:,k].reshape(1,TIME_STEPS)
    
    gasf = GramianAngularField(method='summation')
    x_t_gasf = gasf.fit_transform(f)
     
    gadf = GramianAngularField(method='difference')
    x_t_gadf = gadf.fit_transform(f)
    serie=np.zeros(x_t_gadf.shape)
    diag=normalize_to_zero_one(f)
    # Modificar la diagonal directamente
    print(diag.shape)
    for i in range(0,TIME_STEPS):
        serie[0][i][i]=diag[0][i]
    print(serie)
    X_t_GAF=np.concatenate((x_t_gasf,x_t_gadf,serie),axis=-1)
     
    
    #X_gaf = gramian_angular_field(x)
    #print(X_gaf)
    
    return X_t_GAF

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


def SavevarGAF_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     imx = varGAF(x,'x', TIME_STEPS)
     imy = varGAF(x,'y', TIME_STEPS)
     imz = varGAF(x,'z', TIME_STEPS)

     print("SOLO _r", imx.shape)
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
     img=np.array([imx,imy,imz])
     dim=3
     newImages=[]
     if normalized:
            for i in range(0,dim):
                imagen=img[i]
                _r=imagen[:,:,:TIME_STEPS].reshape(129,129)
                _g=imagen[:,:,TIME_STEPS:TIME_STEPS*2].reshape(129,129)
                _b=imagen[:,:,TIME_STEPS*2:TIME_STEPS*3].reshape(129,129)
               
                newImage = RGBfromMTFMatrix_of_XYZ((_r+1)/2, (_g+1)/2,(_b))
                #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
                # print(newImage.shape)
                #print(newImage[1][4][0]* 255)
                
                newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
                # plt.imshow(newImage)
                newImages=np.append(newImages,newImage)
                if saveImage:
                    # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
                    newImage.save(f"{path}{sj}{action}{item_idx}gaf{i}.png")
                # plt.close('all')
       
    
     return newImages
    
    else:
     return None
    
   

def Reconstruct_GAF(img,dictionary,a):
    _r= img[:,:,0].astype('float') #sumatoria de cosenos
    _g= img[:,:,1].astype('float') #diff de senos
    _b= img[:,:,2].astype('float') #canal libre
    
    _r=np.interp(_r,(0,255),(-1,1))
    _g=np.interp(_g,(0,255),(-1,1))
    print("tamaño de b",_b.shape)
    _b=np.interp(_b,(0,255),(dictionary[a][1][0],dictionary[a][0][0]))
    #print(_r)
    #r=GAF_to_TS(_r,_g,0,dictionary[a])
    #g=GAF_to_TS(_g,1,dictionary[a])
    b=np.diag(_b)
    N=[]
    N.append(b)
    #N.append(_g)
    #N.append(_b)
    return N
def equations(vars, C, S):
    x_i, x_j = vars
    x_i=x_j
    eq1 = x_i * x_i + np.sqrt((1 - x_i**2) * (1 - x_i**2)) - C
    eq2 = x_i * np.sqrt(1 - x_i**2) - x_i * np.sqrt(1 - x_i**2) - S
    return [eq1, eq2]
def GAF_to_TS(_r,_g,i,dictionary,method='summation'):
    #We recontruct first phi vector based on gasf matrix diagonal property
    
    cose=np.diag(_r)
    sen=np.diag(_g)
    X_normalized=[]
    phi=np.arccos(np.diag(_r))/2
    X_normalized=np.cos(phi)
    

    


    
     
    #RECONSTRUCCIÓN DISCRETA DE LOS ESTADOS:
    #Sabemos que el valor medio de los datos maximos es 
    MAX=dictionary[0] 
    MIN=dictionary[1]
    

    #x=np.interp(X_normalized,(np.min(X_normalized),np.max(X_normalized)),(MIN[i],MAX[i]))
    x=X_normalized
       
    return x

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
     a=0
     w = X_train[a]
     sj = sj_train[a][0]
     w_y = y_train[a]
     w_y_no_cat = np.argmax(w_y)
     f=w[:,0].reshape(1,129)
     print(f.shape)
     print(f)
     time_points = np.linspace(0, 4 * np.pi, 129)
     x = np.sin(time_points)
     X = np.array([x])
     print(X.shape)
     """
     gasf= GramianAngularField(sample_range=(-1,1),method='summation')
     x_t_gasf=gasf.fit_transform(normalize_series(f))
     gadf= GramianAngularField(sample_range=(-1,1),method='summation')
     x_t_gadf=gadf.transform(f)
     
     """
    
     img = SavevarGAF_XYZ(w, sj, a, "x", normalized = 1, path=f"./", TIME_STEPS=129)

     
     dictionary=dict()
     for k in range(0,8163):
         l=X_train[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]
    
     imagen = cv2.imread("./1600x0gaf0.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     
     rp=Reconstruct_GAF(imagen,dictionary,a)
     dim=0

   
    # Configurar el estilo de los gráficos
     plt.style.use("ggplot")  

    # Gráfico original
     plt.figure(figsize=(10, 6))
     plt.plot(w[:, dim], marker='', color='blue',cmap="viridis")
     plt.title('Original', fontsize=18,fontweight="bold")
     plt.xlabel("Tiempo", fontsize=12)
     plt.ylabel('Índice X', fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('original.png', bbox_inches='tight', pad_inches=0)
     plt.clf()

    # Gráfico reconstrucción
     plt.figure(figsize=(10, 6))
     plt.plot(rp[dim], marker='', color='magenta')
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
     plt.legend(fontsize=12)
     plt.grid(True)
     plt.tight_layout()
     plt.savefig('Comparativa.png', bbox_inches='tight', pad_inches=0)
     plt.clf()
     
     f=np.array(w[:,dim])
     f=f[:]
     print(f.shape)
     error_absoluto, error_relativo = err.calcular_errores(f, rp[dim])
     #d = dtw.distance_fast(f, rp[1], use_pruning=True)
     print(f"Error Absoluto Promedio: {error_absoluto}")
     print(f"Error Relativo Promedio: {error_relativo}")
     print(f"Error Estandar Promedio: {err.stderror(f,rp[dim])}")
     print(f"Error Cuadrático Promedio: {err.rmse(f,rp[dim])}")
     #print(f"Error DTW: {d}")
     print(f"Coeficiente de correlación: {np.corrcoef(f, rp[dim])[0,1]}")
     

if __name__ == '__main__':
    main()