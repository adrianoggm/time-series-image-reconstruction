#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import tgen.activity_data as act
import tgen.recurrence_plots as rec
from pyts.image import MarkovTransitionField
from scipy.sparse.csgraph import dijkstra
from PIL import Image
import cv2
from sklearn.manifold import MDS
from dtaidistance import dtw
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
       THRESHOLD=RecurrrenceTreshold(rp,20)
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


def calculate_shortest_path_matrix(Wg):
    shortest_path_matrix = dijkstra(Wg, directed=True)
    return shortest_path_matrix + 1e-10


def SavevarRP_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     #print(x.shape)
     _r = rec.varRP(x,'x', TIME_STEPS)
     _g = rec.varRP(x,'y', TIME_STEPS)
     _b = rec.varRP(x,'z', TIME_STEPS)

     
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
          newImage = rec.RGBfromRPMatrix_of_XYZ(NormalizeMatrix_Adri(_r), NormalizeMatrix_Adri(_g), NormalizeMatrix_Adri(_b))
          #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          # print(newImage.shape)
          #print(newImage[1][4][0]* 255)
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
               newImage.save(f"{path}{sj}{action}{item_idx}.png")
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
    #print("X2",_r[1][4])
    ##PREGUNTAR SI ES LO MISMO O HABRIA QUE PASAR de 255 a [0-1] y posteriormente a min max
    _r=np.interp(_r,(0,255),(_min,_max))
    _g=np.interp(_g,(0,255),(_min,_max))
    _b=np.interp(_b,(0,255),(_min,_max))
    #print("X2 post normalizacion",_r[1][4])
    
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
        rp=fix_rotationscale(R[i],rp,i)
        N.append(rp)
    N=np.array(N)
    return N
def fix_rotationscale(rporiginal,seriereconstruida,i,TIME_STEPS=129):
     MAX=[10.657428709640488,3.0590269720681738,7.629156537079175] 
     MIN=[-4.128293834805366,-11.814377181580914,-5.316818145702738]
     MEAN=[3.451300189402157,-4.152375746050367, 1.2525971513133263]
     #escalo la serie para que tenga los valores 
     
     
     #print(seriereconstruida.shape)
     #Scaling part
     n=np.append(seriereconstruida,np.mean(seriereconstruida))
     #n=seriereconstruida
     #print(n.shape)
     posi=np.interp(n,(np.min(n),np.max(n)),(MIN[i],MAX[i])).reshape(129)
     meandiff=MEAN[i]- np.mean(posi[i])    
     posi=posi+meandiff 
     rposi=rec.varRP2(posi, TIME_STEPS=129)    
     
     n=n*-1  
     nega=np.interp(n,(np.min(n),np.max(n)),(MIN[i],MAX[i])).reshape(129)
     meandiff=MEAN[i]- np.mean(nega[i])  
     nega=nega+meandiff  
     rnega=rec.varRP2(nega, TIME_STEPS=129)
     
     rp=[]
     #print(rporiginal.shape,rposi.shape)
     
     error_absolutoa, error_relativoa= calcular_errores(rporiginal, rposi)
     error_absolutob, error_relativob= calcular_errores(rporiginal, rnega)
     if error_relativob<error_relativoa :
         rp=nega[:128]
         #print("nega")
     else :
        rp=posi[:128]
        #print("posi")
        
     return  rp 
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
     w = X_train[0]
     sj = sj_train[0][0]
     w_y = y_train[0]
     w_y_no_cat = np.argmax(w_y)
     print(w.shape)
     img = SavevarRP_XYZ(w, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     #parte de reconstruccion
     #primero genero la RP a partir de la imagen
     imagen = cv2.imread("./1600x0.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     rp=Reconstruct_RP(imagen)
     
     """
     #5493 para el max 
     #
     vmax=[]
     vmin=[]
     vmed=[]
     print(X_train.shape)
     for x in X_train:
         
         vmax.append(np.max(x[:,:]))
         vmin.append(np.min(x[:,:]))
         
     print("media Max",np.max(vmax))
     vmax=np.array(vmax)
     indices = [i for i, x in enumerate(vmax) if x >= 66.60]
     print(indices)
     print("media Min",np.mean(vmin)) 
     print(np.mean(X_train[:,:,0]),np.mean(X_train[:,:,1]),np.mean(X_train[:,:,2]))
     _max=np.mean(vmax)
     _min=np.mean(vmin)
     
     
        
        
     
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
     d = dtw.distance_fast(f, rp[0], use_pruning=True)
     print(f"Error Absoluto Promedio: {error_absoluto}")
     print(f"Error Relativo Promedio: {error_relativo}")
     print(f"Error DTW: {d}")
     print(f"Coeficiente de correlación: {np.corrcoef(f, rp[0])[0,1]}")
     """
     a=0
     b=0 
     c=0 
     s=np.append(s.astype(np.float64),a)
     s1=np.append(s1.astype(np.float64),b)
     s2=np.append(s2.astype(np.float64),c)

     w2=np.zeros(w.shape)
     print(w2.dtype)
     for i in range(0,129):
        w2[i][0]=s[i]
        w2[i][1]=s1[i]
        w2[i][2]=s2[i]
   
    
     img = SavevarRP_XYZ(w2, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129)
    #Error Absoluto Promedio: 1.2916679603210046
    #Error Relativo Promedio: 0.10410946116987463
    #Coeficiente de correlación: 0.9152496948611255
    """
     
     
# Guardar el gráfico como una imagen
    

# Mostrar el gráfico (opcional)
plt.show()
if __name__ == '__main__':
    main()