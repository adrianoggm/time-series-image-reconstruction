import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import math
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.manifold import MDS
from PIL import Image
from scipy.sparse.csgraph import dijkstra
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
def varRP2(data, TIME_STEPS=129):#dim:=x,y,z
    x = data
    
    
    s = []
    for i in range(len(x)-1):
    # for i in range(len(x)):

        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
    
        
    #print s
    # dimR = len(x)-1
    dimR = len(s)

    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            # if i==0 and j==0:
              # print("s[i]", s[i], "s[j]", s[j])
              # print(list(zip(s[i], s[j])))
              # print(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))))
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
            # R[i][j] = Distance2dim(s[i],s[j])
    return R
def vector_magnitude(vector):
    x, y = vector  # unpack the vector into its two components
    magnitude = math.sqrt(x**2 + y**2)  # calculate the magnitude using the formula
    return magnitude
def Distance2dim(a,b):
    return pow(pow(float(a[1])-float(b[1]),2)+pow(float(a[0])-float(b[0]),2), 0.5)
def Cosin2vec(a,b):
    # print("a", a)
    # print("b", b)
    x = a[1]*b[1]+a[0]*b[0]
    y = (pow(pow(float(a[1]),2) + pow(float(a[0]),2) , 0.5) * pow(pow(float(b[1]),2) + pow(float(b[0]),2) , 0.5)) 
    return  x / y if y>0 else 0
def WeightAngle(a,b):
    return math.exp(2*(1.1 - Cosin2vec(a,b)))
def varRP_axis(data, axis,  TIME_STEPS=129):#dim:=x,y,z
    x = []
    
    for j in range(TIME_STEPS):
        x.append(data[j][axis])
    
    s = []
    for i in range(len(x)-1):
        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    # dimR = len(x)-1
    dimR = len(s)
    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
    return R
def varRP(data, dim, TIME_STEPS=129):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(TIME_STEPS):
            # print("dato:", data[j][0])
            x.append(data[j][0])
    elif dim == 'y':
        for j in range(TIME_STEPS):
            x.append(data[j][1])
    elif dim == 'z':
        for j in range(TIME_STEPS):
            x.append(data[j][2])
    
    s = []
    for i in range(len(x)-1):
    # for i in range(len(x)):

        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
    
        
    #print s
    # dimR = len(x)-1
    dimR = len(s)

    #R = np.zeros((dimR,dimR))
    R = np.eye(dimR)
    for i in range(dimR):
        for j in range(dimR):
            # if i==0 and j==0:
              # print("s[i]", s[i], "s[j]", s[j])
              # print(list(zip(s[i], s[j])))
              # print(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))))
            if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
                sign =1.0
            else:
                sign =-1.0
            R[i][j] = sign*Distance2dim(s[i],s[j])
            # R[i][j] = Distance2dim(s[i],s[j])
    return R
def RP(data, dim, TIME_STEPS=129):#dim:=x,y,z
    x = []
    if dim == 'x':
        for j in range(TIME_STEPS):
            x.append(data[j][0])
    elif dim == 'y':
        for j in range(TIME_STEPS):
            x.append(data[j][1])
    elif dim == 'z':
        for j in range(TIME_STEPS):
            x.append(data[j][2])
    
    s = []
    for i in range(len(x)-1):
    # for i in range(len(x)):

        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    # dimR = len(x)-1
    dimR = len(s)
    R = np.zeros((dimR,dimR))

    for i in range(dimR):
        for j in range(dimR):
            R[i][j] = Distance2dim(s[i],s[j])
    return R
def RP_axis(data, axis, th=None, TIME_STEPS=129):#dim:=x,y,z
    x = []
    
    for j in range(TIME_STEPS):
        x.append(data[j][axis])
    
    s = []
    for i in range(len(x)-1):
    # for i in range(len(x)):

        _s = []
        _s.append(x[i])
        _s.append(x[i+1])
        s.append(_s)
        
    #print s
    # dimR = len(x)-1
    dimR = len(s)

    R = np.zeros((dimR,dimR))

    how_plot = 0
    for i in range(dimR):
        for j in range(dimR):

          # print(Distance2dim(s[i],s[j]))

          if th == None:
            R[i][j] = Distance2dim(s[i],s[j])
            how_plot += 1
          else:
            R[i][j] = 1 if Distance2dim(s[i],s[j]) <= th else 0
            if R[i][j] == 1:
              how_plot += 1
    print(f"plotting {(how_plot/(dimR*dimR))*100}%")
    return R
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
def RGBfromRPMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    
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
def RGBfromRPMatrix_of_single_axis(X):   
    dimImage = X.shape[0]
    newImage = np.zeros((dimImage,dimImage,3))
    for i in range(dimImage):
        for j in range(dimImage):
            _pixel = []
            # print(X[i][j])
            _pixel.append(X[i][j])
            # _pixel.append(Y[i][j])
            # _pixel.append(Z[i][j])
            newImage[i][j] = _pixel
    return newImage
def SaveRP(x_array,y_array,z_array, TIME_STEPS=129):
    _r = RP(x_array, "x", TIME_STEPS)
    _g = RP(y_array, "y", TIME_STEPS)
    _b = RP(z_array, "z", TIME_STEPS)
    plt.close('all')
    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
    plt.imshow(newImage)
    # plt.savefig('D:\Datasets\ADL_Dataset\\'+action+'\\'+'RP\\''{}{}.png' .format(action, subject[15:]),bbox_inches='tight',pad_inches = 0)
    plt.close('all')
def SaveRP_XYZ(x, sj, item_idx, action, normalized, path, saveImage=True, TIME_STEPS=129):
    _r = RP(x,'x', TIME_STEPS)
    _g = RP(x,'y', TIME_STEPS)
    _b = RP(x,'z', TIME_STEPS)
    # plt.close('all')
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))#newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)

        newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))

        # plt.imshow(newImage)
        if saveImage:
          # plt.savefig(f"{path}{sj}{action}{item_idx}_rp.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
          newImage.save(f"{path}{sj}{action}{item_idx}_rp.png")
        # plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))

        # plt.imshow(newImage)
        if saveImage:
          # plt.savefig(f"{path}{sj}{action}{item_idx}_rp.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
          newImage.save(f"{path}{sj}{action}{item_idx}_rp.png")

        # plt.close('all')
    return newImage

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

     if normalized:
          newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
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
          newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}.png")
          # plt.close('all')
     return newImage
    else:
     return None
  
def SavevarRP_fran(x, axis, sj=0, item_idx=0, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    _r = varRP_axis(x, axis, TIME_STEPS) 
    # _g = varRP(x,'x') #np.full((_b.shape[0], _b.shape[1]), 255) #varRP(x,'x')
    # _b = varRP(x,'x')
    # _r = np.full((_b.shape[0], _b.shape[1]), 255) #varRP(x,'x') #np.zeros((_r.shape[0], _r.shape[1]))

    # _g = varRP(x,'y')
    # _b = varRP(x,'z')
    plt.close('all')

    plt.axis('off')
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    print("fig size: width=", plt.figure().get_figwidth(), "height=", plt.figure().get_figheight())
    if normalized:
        newImage = NormalizeMatrix(_r) #RGBfromRPMatrix_of_single_axis(NormalizeMatrix(_r))
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))

        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        # plt.imshow(newImage, cmap="viridis")
        if saveImage:
          newImage.save(f"{path}{sj}{action}_{axis}_{item_idx}.png")
          # plt.savefig(f"{path}{sj}{action}_{axis}_{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
        # plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_single_axis(_r)
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))

        # plt.imshow(newImage, cmap="viridis")
        if saveImage:
          newImage.save(f"{path}{sj}{action}_{axis}_{item_idx}.png")
          # plt.savefig(f"{path}{sj}{action}_{axis}_{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
        # plt.close('all')
    return newImage
def SaveRP_fran(x, axis, sj=0, item_idx=0, action=None, normalized=True, path=None, saveImage=True, th=None, TIME_STEPS=129):
    _r = RP_axis(x, axis, th, TIME_STEPS) 
    _g = RP_axis(x, axis, th, TIME_STEPS) 
    _b = RP_axis(x, axis, th, TIME_STEPS) 
    
    # plt.close('all')
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    
    if normalized:
        # newImage = NormalizeMatrix(_r) #RGBfromRPMatrix_of_single_axis(NormalizeMatrix(_r))
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))

        # plt.imshow(newImage, cmap="viridis")
        if saveImage:
          newImage.save(f"{path}{sj}{action}_{axis}_{item_idx}.png")
          # plt.savefig(f"{path}{sj}{action}_{axis}_{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
        # plt.close('all')
    else:
        # newImage = RGBfromRPMatrix_of_single_axis(_r)
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))

        # plt.imshow(newImage, cmap="viridis")
        if saveImage:
          newImage.save(f"{path}{sj}{action}_{axis}_{item_idx}.png")
          # plt.savefig(f"{path}{sj}{action}_{axis}_{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
        # plt.close('all')
    return newImage

def RecurrrenceTreshold(rp,X):
    distances=np.abs(rp)
    epsilon_X=np.percentile(distances,X)
    return epsilon_X

def FuncionC(rp,cord1,cord2,method="Euclid"):
    if(method=="Euclid"):
       THRESHOLD=RecurrrenceTreshold(rp,75) # THRESHOLD=RecurrrenceTreshold(rp,75)
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
     rposi=varRP2(posi, TIME_STEPS)    
     
     n=n*-1+1  
     min_a, max_a = np.min(n), np.max(n)  
     
     #nega = min_b + n * (max_b - min_b)
     nega=np.interp(n,(min_a,max_a),(MIN[i], MAX[i]))
     rnega=varRP2(nega, TIME_STEPS)
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
         rp=nega[:128]
         inverted+=1
     else :
        rp=posi[:128]
     
        #print("posi")
        
     return  rp,inverted

    