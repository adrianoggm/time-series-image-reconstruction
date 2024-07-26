#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

#import tgen.activity_data as act
import activity_data as act
#import tgen.recurrence_plots as rec
import recurrence_plots as rec
#import tgen.calculate_ts_errors as err
import calculate_ts_errors as err
from pyts.image import MarkovTransitionField
from scipy.sparse.csgraph import dijkstra
from PIL import Image
import cv2
from sklearn.manifold import MDS
from dtaidistance import dtw
import sklearn.metrics as metrics
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
def NormalizeMatrix_Adri(_r):
    dimR = _r.shape[0]
    _max=66.615074
    _min =  -78.47761
    _max_min = _max - _min
    _normalizedRP=np.interp(_r,(_min,_max),(0,1))
    """
    _normalizedRP = np.zeros((dimR,dimR))
    for i in range(dimR):
        for j in range(dimR):
            _normalizedRP[i][j] = (_r[i][j]-_min)/_max_min
    """
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
       THRESHOLD=RecurrrenceTreshold(rp,30) # THRESHOLD=RecurrrenceTreshold(rp,75)
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
def Reconstruct_RP(img,dictionary,valor):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    #Obtengo cada una de las recurrence plots
    
    #print("X2",_r[1][4])
    ##PREGUNTAR SI ES LO MISMO O HABRIA QUE PASAR de 255 a [0-1] y posteriormente a min max
    _r=np.interp(_r,(0,255),(dictionary[valor][1][0],dictionary[valor][0][0]))
    _g=np.interp(_g,(0,255),(dictionary[valor][1][1],dictionary[valor][0][1]))
    _b=np.interp(_b,(0,255),(dictionary[valor][1][2],dictionary[valor][0][2]))
    
    
    R= []
    R.append(_r)
    R.append(_g)
    R.append(_b)
    n=len(R)
    N=[]
    ninversiones=0
    for i in range(0,n):
        wg=weigthed_graphRP(R[i]) 
        spm=calculate_shortest_path_matrix(wg)  
        ##we multiply * -1 ya que se invierte al calcular los valores propios
        rp=reconstruct_time_series(spm, ep=0.0)
        rp,a=fix_rotationscale(R[i],rp,i,dictionary[valor])
        ninversiones=ninversiones+a
        N.append(rp)
    N=np.array(N)
    return N,ninversiones
def fix_rotationscale(rporiginal,seriereconstruida,i,dictionary,TIME_STEPS=129):
     #print(dictionary)
     MAX=dictionary[0]
     MIN=dictionary[1]
     
     #escalo la serie para que tenga los valores 
     
     
     #n=(seriereconstruida-min_val)/(max_val-min_val)
     #print(seriereconstruida.shape)
     #Scaling part
     #n=seriereconstruida
     n=np.append(seriereconstruida,np.mean(seriereconstruida))
     
     #posi= MIN[i]+n*(MAX[i]-1)
     min_a, max_a = np.min(n), np.max(n)  
     

    # Transformar los puntos
     #posi =min_b + n * (max_b - min_b)
     posi=np.interp(n,(min_a,max_a),(MIN[i], MAX[i]))
     rposi=rec.varRP2(posi, TIME_STEPS)    
     
     n=n*-1+1  
     min_a, max_a = np.min(n), np.max(n)  
     
     #nega = min_b + n * (max_b - min_b)
     nega=np.interp(n,(min_a,max_a),(MIN[i], MAX[i]))
     rnega=rec.varRP2(nega, TIME_STEPS)
     #print(n)
     #print(nega)

     rp=[]
     #print(rporiginal.shape,rposi.shape)
     
     #error_absolutoa, error_relativoa= calcular_errores(rporiginal[:60], rposi[0])
     #error_absolutob, error_relativob= calcular_errores(rporiginal[:60], rnega[0])
     distancia_euclidianaa = np.linalg.norm(rporiginal - rposi)
     distancia_euclidianab = np.linalg.norm(rporiginal - rnega)
     inverted=0
     if distancia_euclidianab<distancia_euclidianaa :
         rp=nega[:129]
         inverted+=1
     else :
        rp=posi[:129]
     
        #print("posi")
        
     return  rp,inverted


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
     img = rec.SavevarRP_XYZ(w, sj, 5, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     #parte de reconstruccion
     #primero genero la RP a partir de la imagen
     imagen = cv2.imread("./1600x02.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     rp,ninv=Reconstruct_RP(imagen,dictionary,a)
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
     f=f[:]
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