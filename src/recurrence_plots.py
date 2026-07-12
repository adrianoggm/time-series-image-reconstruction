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
def varRP2(data, TIME_STEPS=128):#dim:=x,y,z
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
def varRP_axis(data, axis,  TIME_STEPS=128):#dim:=x,y,z
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
def varRP(data, dim, TIME_STEPS=128):#dim:=x,y,z
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
def RP(data, dim, TIME_STEPS=128):#dim:=x,y,z
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
def RP_axis(data, axis, th=None, TIME_STEPS=128):#dim:=x,y,z
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
def SaveRP(x_array,y_array,z_array, TIME_STEPS=128):
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
def SaveRP_XYZ(x, sj, item_idx, action, normalized, path, saveImage=True, TIME_STEPS=128):
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

def SavevarRP_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=128):
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
  
def SavevarRP_fran(x, axis, sj=0, item_idx=0, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=128):
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
def SaveRP_fran(x, axis, sj=0, item_idx=0, action=None, normalized=True, path=None, saveImage=True, th=None, TIME_STEPS=128):
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
    # Kept for backwards compatibility with any caller that scalar-tests a
    # single (i, j). The hot loop in `CreateCostMatrix` no longer uses it.
    if(method=="Euclid"):
       THRESHOLD=RecurrrenceTreshold(rp,75)
       d=rp[cord1][cord2]
       return 1 if (-THRESHOLD <= d <= THRESHOLD) else 0
    return 0

def CreateCostMatrix(rp, percentile: float = 75.0):
    """Vectorised cost matrix: 1 where |rp[i,j]| <= percentile-threshold, else 0.

    Replaces the original O(N^4 log N) implementation (an N^2 Python loop where
    every cell recomputed `np.percentile` over the full matrix). The new code
    computes the threshold once and uses a single boolean comparison over the
    array, dropping the cost to a single O(N^2 log N) pass driven by C-level
    NumPy ops.
    """
    threshold = np.percentile(np.abs(rp), percentile)
    return (np.abs(rp) <= threshold).astype(rp.dtype, copy=False)



def calculate_shortest_path_matrix(Wg):
    shortest_path_matrix = dijkstra(Wg, directed=True)
    return shortest_path_matrix + 1e-10



"""
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
"""
def weigthed_graphRP(rp, percentile=75.0):
    CostM = CreateCostMatrix(rp, percentile)
    # Sumas por fila: número de elementos "1" en cada fila
    row_sums = np.sum(CostM, axis=1, keepdims=True)  # forma (N, 1)
    # Intersección: producto punto entre CostM y su transpuesta
    intersections = np.dot(CostM, CostM.T)
    # Unión: suma de elementos de ambas filas menos la intersección
    unions = row_sums + row_sums.T - intersections
    # Evitar división por cero
    unions[unions == 0] = 1
    weighted_adjacency_matrix = 1 - (intersections / unions)
    return weighted_adjacency_matrix   
def _classical_mds(D: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Closed-form classical (Torgerson) MDS for a precomputed distance matrix.

    For Euclidean-compatible D this is the exact optimum that SMACOF only
    approximates by iteration. Replaces sklearn's 300-iter SMACOF in the
    inner loop of RP reconstruction (~30x faster per window).
    """
    n = D.shape[0]
    # Double-centering: B = -1/2 J D^2 J  with J = I - 1/n 11^T
    D2 = np.square(D)
    J = np.eye(n) - np.full((n, n), 1.0 / n)
    B = -0.5 * J @ D2 @ J
    # B is symmetric → use eigh (returns ascending eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(B)
    # Pick the top n_components positive eigenvalues
    idx = np.argsort(eigvals)[::-1][:n_components]
    L = np.clip(eigvals[idx], a_min=0.0, a_max=None)
    return eigvecs[:, idx] * np.sqrt(L)


def reconstruct_time_series(shortest_path_matrix, ep=0.0, small_constant=1e-10):
    # Symmetrise + sanitise the input distance matrix.
    symmetric_shortest_path_matrix = 0.5 * (shortest_path_matrix + shortest_path_matrix.T)
    finite_shortest_path_matrix = np.where(
        np.isfinite(symmetric_shortest_path_matrix),
        symmetric_shortest_path_matrix,
        ep,
    )
    finite_shortest_path_matrix += small_constant

    # Closed-form MDS in place of SMACOF — exact, deterministic, ~30x faster.
    embedded_coords = _classical_mds(finite_shortest_path_matrix, n_components=2)

    # Pick the principal axis (largest variance) — mirrors the original code
    # which used cov + eig + argmax.
    variances = np.var(embedded_coords, axis=0)
    max_eigenvalue_index = int(np.argmax(variances))
    return embedded_coords[:, max_eigenvalue_index][:, np.newaxis]


def reconstruct_state_embedding(shortest_path_matrix, ep=0.0, small_constant=1e-10):
    """Same front-end as ``reconstruct_time_series`` but keep BOTH MDS axes.

    Returns the (N-1, 2) embedding of the recovered adjacent-states instead of
    collapsing each state onto its largest-eigenvalue axis. This is the 2-D
    decoder of Section~sec:rp_reconstruction_method (the principled variant that
    exploits the state overlap); the 1-D projection above stays the default.
    """
    symmetric = 0.5 * (shortest_path_matrix + shortest_path_matrix.T)
    finite = np.where(np.isfinite(symmetric), symmetric, ep) + small_constant
    return _classical_mds(finite, n_components=2)


def reconstruct_from_embedding_2d(embedding, n_theta=720):
    """Recover an N-sample signal from the (N-1, 2) state embedding.

    Aligns axis-2 of state i with axis-1 of state i+1 (the RP overlap condition
    s_i=(x_i,x_{i+1})) over the orthogonal group O(2), then averages the two
    estimates of each shared sample. Returns a length-N series *before* amplitude
    rescaling and *before* the global-sign choice; both are handled downstream by
    ``fix_rotationscale(..., append_value="none")``, exactly as for the 1-D path.
    """
    P = np.asarray(embedding, dtype=float)  # (M, 2), M = N-1 states
    M = P.shape[0]

    # Best R in O(2) minimising sum_i ((R p_i)_2 - (R p_{i+1})_1)^2. The true
    # configuration satisfies the overlap exactly, so its aligning transform is
    # the global minimum (up to a sign flip, resolved downstream by signed RP).
    # ponytail: theta grid scan (O(n_theta)); J(theta) is a 2-theta sinusoid so a
    # closed form exists — swap it in only if this ever shows up in a profile.
    thetas = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    best_J, best_R = None, None
    for reflect in (1.0, -1.0):
        for th in thetas:
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, -s * reflect], [s, c * reflect]])  # rot(th)·diag(1,reflect) spans O(2)
            S = P @ R.T
            J = float(np.sum((S[:-1, 1] - S[1:, 0]) ** 2))
            if best_J is None or J < best_J:
                best_J, best_R = J, R
    S = P @ best_R.T

    # Overlap reconstruction: x1 = s[0,0]; x_{i+1} = mean(s[i,1], s[i+1,0]); xN = s[M-1,1].
    x = np.empty(M + 1)
    x[0] = S[0, 0]
    x[1:M] = 0.5 * (S[:-1, 1] + S[1:, 0])
    x[M] = S[M - 1, 1]
    return x


def Reconstruct_RP(img,dictionary,valor,percentile=75.0,method="1d",rp_range=None):
    if method == "direct2ds":
        # Validated DIRECT-2D-S decoder (see src/rp_decoders.py): decodes the RP
        # input with its OWN matrix range (rp_range) instead of the signal
        # amplitude range, avoiding the affine distortion the legacy path below
        # applies to the distances handed to MDS.
        from src.rp_decoders import decode_window
        rr = rp_range[valor] if rp_range is not None else None
        rec_tc, _ = decode_window(img, dictionary[valor], source="direct", dims=2,
                                  recover="overlap", signed=True, rho=percentile / 100.0,
                                  rp_range=rr)
        return rec_tc.T, 0   # (C, T) to match the stacking convention; 0 = inversions n/a
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
        wg=weigthed_graphRP(R[i], percentile)
        spm=calculate_shortest_path_matrix(wg)
        ##we multiply * -1 ya que se invierte al calcular los valores propios
        if method == "2d":
            # Principled 2-D decoder: keep both MDS axes, align on the state
            # overlap, average the shared samples (already length N, so no append).
            emb = reconstruct_state_embedding(spm, ep=0.0)
            rp = reconstruct_from_embedding_2d(emb)
            rp, a = fix_rotationscale(R[i], rp, i, dictionary[valor], append_value="none")
        else:
            rp=reconstruct_time_series(spm, ep=0.0)
            rp,a=fix_rotationscale(R[i],rp,i,dictionary[valor])
        ninversiones=ninversiones+a
        N.append(rp)
    N=np.array(N)
    return N,ninversiones
def fix_rotationscale(rporiginal,seriereconstruida,i,dictionary,TIME_STEPS=128,
                      use_orientation=True, append_value="mean"):
     # ``use_orientation`` / ``append_value`` expose the two RP post-processing
     # choices for ablation (Section sec:rp_ablation); the defaults reproduce the
     # reported decoder exactly (signed orientation check on, per-window mean).
     MAX=dictionary[0]
     MIN=dictionary[1]

     # Re-augment the N-1 adjacent-state series back to N samples.
     if append_value == "mean":
         n = np.append(seriereconstruida, np.mean(seriereconstruida))
     elif append_value == "endpoint":
         n = np.append(seriereconstruida, seriereconstruida[-1])
     else:
         n = np.asarray(seriereconstruida)

     min_a, max_a = np.min(n), np.max(n)
     posi=np.interp(n,(min_a,max_a),(MIN[i], MAX[i]))
     rposi=varRP2(posi, TIME_STEPS)

     inverted = 0
     if not use_orientation:
         # Ablation: keep the direct MDS orientation, no signed-RP disambiguation.
         rp = posi[:TIME_STEPS]
     else:
         n = n*-1+1
         min_a, max_a = np.min(n), np.max(n)
         nega=np.interp(n,(min_a,max_a),(MIN[i], MAX[i]))
         rnega=varRP2(nega, TIME_STEPS)
         # Pick the orientation (positive or negated) whose RP image is closer to
         # the encoded one in Euclidean distance.
         distancia_euclidianaa = np.linalg.norm(rporiginal - rposi)
         distancia_euclidianab = np.linalg.norm(rporiginal - rnega)
         if distancia_euclidianab < distancia_euclidianaa:
             rp = nega[:TIME_STEPS]
             inverted += 1
         else:
             rp = posi[:TIME_STEPS]

     # `posi` already has length TIME_STEPS (= seriereconstruida + 1 appended mean),
     # so the slice here is a no-op — it stays defensive against future changes
     # but does NOT truncate to 128 any more, which used to break parity with
     # GAF/GADF/MTF (all of them return TIME_STEPS samples per channel).
     return rp, inverted

    