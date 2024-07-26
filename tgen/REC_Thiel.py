#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import activity_data as act
import recurrence_plots as rec
from pyts.image import RecurrencePlot
from scipy.sparse.csgraph import dijkstra
from PIL import Image
import cv2
from sklearn.manifold import MDS

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
def DesNormalizeMatrix_Adri(_r):
    dimR = _r.shape[0]
    _max=66.615074
    _min =  -78.47761
    _max_min = _max - _min
    
    
    _desnormalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _desnormalizedRP[i][j] = ((_r[i][j]*_max_min)+_min)
    
    return _desnormalizedRP

def RecurrrenceTreshold(rp,X):
    distances=np.abs(rp)
    epsilon_X=np.percentile(distances,X)
    return epsilon_X

def FuncionC(rp,cord1,cord2,method="Euclid"):
    if(method=="Euclid"):
       THRESHOLD=RecurrrenceTreshold(rp,25)
       valor=0
       d=rp[cord1][cord2]
       if((d<=THRESHOLD)and(d>=(-THRESHOLD))):
          valor=1
    return valor  
def CreateCostMatrix(rp) :
    CostM=np.zeros_like(rp)
    N,M=CostM.shape
    for i in range(N):
         for j in range(M):
             CostM[i][j]=FuncionC(rp,i,j,"Euclid")
    return CostM
def Sort_RP(rp):
    R=CreateCostMatrix(rp)
    #CostM=rp
    n = len(R)

    # Inicializar una lista para almacenar los valores n_{i,j}
    ni_j = np.zeros((n, n), dtype=int)

    # Iterar sobre todos los puntos i
    for i in range(len(R)):
        # Iterar sobre todos los puntos j
        for j in range(len(R)):
            if R[i][j] == 1:  # Si x_i y x_j son vecinos
                # Encontrar los vecinos de x_i
                neighbors_xi = set(k for k in range(len(R)) if R[i][k] == 1)
                # Encontrar los vecinos de x_j
                neighbors_xj = set(k for k in range(len(R)) if R[j][k] == 1)
                # Contar los vecinos de x_j que no son vecinos de x_i
                ni_j[i][j] = len(neighbors_xj - neighbors_xi)
               
                
    # Identificar los puntos x_{j1} y x_{j2}
    candidates = []
    for j in range(n):
     if all(ni_j[i, j] == 0 for i in range(n)):
        candidates.append(j)
    print("Candidatos de cada 1",candidates)

def calculate_shortest_path_matrix(Wg):
    shortest_path_matrix = dijkstra(Wg, directed=True)
    return shortest_path_matrix + 1e-10
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

def varRP1(data,dim, TIME_STEPS):
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z': 
        k=2
    
    
    f=data[:,k].reshape(1,TIME_STEPS)
    
    rp = RecurrencePlot()
    x_t_rp= rp.fit_transform(f)
     
    
    
    serie=np.zeros(x_t_rp.shape)
    diag=normalize_to_zero_one(f)
    # Modificar la diagonal directamente
    print(diag.shape)
    for i in range(0,TIME_STEPS):
        serie[0][i][i]=diag[0][i]
    print(serie)
    X_t_REC=np.concatenate((x_t_rp,serie,serie),axis=-1)
     
    
    #X_gaf = gramian_angular_field(x)
    #print(X_gaf)
    
    return X_t_REC 

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

def SavevarRP1_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     imx = varRP1(x,'x', TIME_STEPS)
     imy = varRP1(x,'y', TIME_STEPS)
     imz = varRP1(x,'z', TIME_STEPS)

     
     
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
                print("_R", _r.shape)
                #_r=np.interp(_b,(min(_r),max(_r)),1)
                newImage = RGBfromMTFMatrix_of_XYZ(rec.NormalizeMatrix(_r), _g,_b)
                #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
                # print(newImage.shape)
                #print(newImage[1][4][0]* 255)
                
                newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
                # plt.imshow(newImage)
                newImages=np.append(newImages,newImage)
                if saveImage:
                    # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
                    newImage.save(f"{path}{sj}{action}{item_idx}RP{i}.png")
                # plt.close('all')
       
    
     return newImages
    
    else:
     return None
    
def varRP(data,dim, TIME_STEPS):
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z': 
        k=2
    
    
    f=data[:,k].reshape(1,TIME_STEPS)
    
    rp = RecurrencePlot()
    x_t_rp= rp.fit_transform(f)
     
    
    
    
     
    
    #X_gaf = gramian_angular_field(x)
    #print(X_gaf)
    
    return x_t_rp[0]

def SavevarRP_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     #print(x.shape)
     _r = varRP(x,'x', TIME_STEPS)
     _g = varRP(x,'y', TIME_STEPS)
     _b = varRP(x,'z', TIME_STEPS)

     
     #print("Y", _g)
     #print("Z", _b)
     
     # plt.close('all')
     # plt.figure(figsize=(1,1))
     # plt.axis('off')
     # plt.margins(0,0)
     # plt.gca().xaxis.set_major_locator(plt.NullLocator())
     # plt.gca().yaxis.set_major_locator(plt.NullLocator())

     #print("fig size: width=", plt.figure().get_figwidth(), "height=", plt.figure().get_figheight())
     print("FORMA",_r.shape)
     if normalized:
          newImage = rec.RGBfromRPMatrix_of_XYZ(rec.NormalizeMatrix(_r), rec.NormalizeMatrix(_g), rec.NormalizeMatrix(_b))
          #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          # print(newImage.shape)
          #print(newImage[1][4][0]* 255)
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
               newImage.save(f"{path}{sj}{action}{item_idx}2.png")
          # plt.close('all')
     else:
          newImage = rec.RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}.png")
          # plt.close('all')
     return newImage
    else:
     return None    
def weigthed_graphRP(rp):
    Sort_RP(rp)
    CostM=CreateCostMatrix(rp)
    weighted_adjacency_matrix = np.zeros_like(rp) 
    nonzero_indices = np.nonzero(CostM)

    for i, j in zip(*nonzero_indices):
        Gi = np.nonzero(CostM[i])[0]
        Gj = np.nonzero(CostM[j])[0]
        intersection = len(np.intersect1d(Gi, Gj))
        union = len(np.union1d(Gi, Gj))
        weighted_adjacency_matrix[i, j] = 1 - (intersection / union)
                
    return weighted_adjacency_matrix
    
def reconstruct_time_series(shortest_path_matrix, ep=0.0, small_constant=1e-10):
    # Forzar la simetría en la matriz
    symmetric_shortest_path_matrix = 0.5 * (shortest_path_matrix + shortest_path_matrix.T)

    # Reemplazar los infinitos en la matriz con un valor grande pero finito
    finite_shortest_path_matrix = np.where(
        np.isfinite(symmetric_shortest_path_matrix),
        symmetric_shortest_path_matrix,
        ep  # Ajusta este valor según sea necesario
    )

    #print("mfinita",finite_shortest_path_matrix)

     # Add a small constant to avoid division by zero
    finite_shortest_path_matrix += small_constant    

    # Apply MDS to get a 2D representation of the shortest path matrix
    mds = MDS(n_components=2, dissimilarity='precomputed',random_state=1)
    embedded_coords = mds.fit_transform(finite_shortest_path_matrix)
    
    #----
    # Calcular la matriz de covarianza
    cov_matrix = np.cov(embedded_coords, rowvar=False)
    #print("mds",embedded_coords)
    # Calcular los valores y vectores propios de la matriz de covarianza
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Encontrar el índice del valor propio más grande
    max_eigenvalue_index = np.argmax(eigenvalues)

    # Seleccionar la columna correspondiente al valor propio más grande en la matriz embedded_coords
    selected_column = embedded_coords[:, max_eigenvalue_index][:, np.newaxis]

    return selected_column
def Reconstruct_RP(img):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    #Obtengo cada una de las recurrence plots
    _max=66.615074
    _min =  -78.47761
    print("X2",_r[1][4])
    ##PREGUNTAR SI ES LO MISMO O HABRIA QUE PASAR de 255 a [0-1] y posteriormente a min max
    _r=np.interp(_r,(0,255),(_min,_max))
    _g=np.interp(_g,(0,255),(_min,_max))
    _b=np.interp(_b,(0,255),(_min,_max))
    print("X2 post normalizacion",_r[1][4])
    
    R= []
    R.append(_r)
    R.append(_g)
    R.append(_b)
    n=len(R)
    N=[]
    for i in range(0,n):
        wg=weigthed_graphRP(R[i]) 
        spm=calculate_shortest_path_matrix(wg)  
        ##we multiply * -1 ya que se invierte al calcular los valores propios
        rp=reconstruct_time_series(spm, ep=0.0)
        
        N.append(rp)
    return N

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
     """
    X_train = np.load("./data/WISDM/numpies/train/windowed/0/1600/x.npy")
    print(X_train.shape)
    # Create a toy time series using the sine function
    
    time_points = X_train[0]
    
    x = np.sin(time_points[0])
    X = np.array([x])
    time_points=np.transpose(time_points,(1,0))
    # Compute Gramian angular fields
    mtf = MarkovTransitionField(n_bins=8)
    print(time_points.shape)
    X_mtf = mtf.fit_transform(X)
    
    

    # Plot the time series and its Markov transition field
    width_ratios = (2, 7, 0.4)
    height_ratios = (2, 7)
    width = 6
    height = width * sum(height_ratios) / sum(width_ratios)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 3,  width_ratios=width_ratios,
                        height_ratios=height_ratios,
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    # Define the ticks and their labels for both axes
    time_ticks = np.linspace(0, 4 * np.pi, 9)
    time_ticklabels = [r'$0$', r'$\frac{\pi}{2}$', r'$\pi$',
                    r'$\frac{3\pi}{2}$', r'$2\pi$', r'$\frac{5\pi}{2}$',
                    r'$3\pi$', r'$\frac{7\pi}{2}$', r'$4\pi$']
    value_ticks = [-1, 0, 1]
    reversed_value_ticks = value_ticks[::-1]

    # Plot the time series on the left with inverted axes
    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.plot(x, time_points)
    ax_left.set_xticks(reversed_value_ticks)
    ax_left.set_xticklabels(reversed_value_ticks, rotation=90)
    ax_left.set_yticks(time_ticks)
    ax_left.set_yticklabels(time_ticklabels, rotation=90)
    ax_left.set_ylim((0, 4 * np.pi))
    ax_left.invert_xaxis()

    # Plot the time series on the top
    ax_top = fig.add_subplot(gs[0, 1])
    ax_top.plot(time_points, x)
    ax_top.set_xticks(time_ticks)
    ax_top.set_xticklabels(time_ticklabels)
    ax_top.set_yticks(value_ticks)
    ax_top.set_yticklabels(value_ticks)
    ax_top.xaxis.tick_top()
    ax_top.set_xlim((0, 4 * np.pi))
    ax_top.set_yticklabels(value_ticks)

    # Plot the Gramian angular fields on the bottom right
    ax_mtf = fig.add_subplot(gs[1, 1])
    im = ax_mtf.imshow(X_mtf, cmap='rainbow', origin='lower', vmin=0., vmax=1.,
                    extent=[0, 4 * np.pi, 0, 4 * np.pi])
    ax_mtf.set_xticks([])
    ax_mtf.set_yticks([])
    ax_mtf.set_title('Markov Transition Field', y=-0.09)

    # Add colorbar
    ax_cbar = fig.add_subplot(gs[1, 2])
    fig.colorbar(im, cax=ax_cbar)

    plt.show()
    plt.savefig("imagen.png")
   """
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
     img = SavevarRP_XYZ(w, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     #parte de reconstruccion
     #primero genero la RP a partir de la imagen
     imagen = cv2.imread("./1600x0.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     


    
       
# Guardar el gráfico como una imagen
    

# Mostrar el gráfico (opcional)
plt.show()
if __name__ == '__main__':
    main()