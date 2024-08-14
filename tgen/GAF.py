#!/home/adriano/Escritorio/TFG/venv/bin/python3
import math
import numpy as np
import matplotlib.pyplot as plt
import tgen.activity_data as act
import tgen.recurrence_plots as rec
#import activity_data as act
#import recurrence_plots as rec
from PIL import Image
import cv2
import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
import os
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
    normalized_series = normalize_to_zero_one(series)
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
        print("entra")
    elif method == 'difference':
        gaf = tabulate(phi, phi, sin_diff)
    else:
        raise ValueError("Method must be 'summation' or 'difference'")
    print(gaf)
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
    
    x=np.array(data[:,k]).reshape(1,-1)
    
    X_gaf = gramian_angular_field(x)
    #print(X_gaf)
    
    return X_gaf

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
     _r = varGAF(x,'x', TIME_STEPS)
     _g = varGAF(x,'y', TIME_STEPS)
     _b = varGAF(x,'z', TIME_STEPS)

     
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
          print(_r)
          newImage = RGBfromMTFMatrix_of_XYZ(normalize_to_zero_one(_r), normalize_to_zero_one(_g),normalize_to_zero_one(_b))
          #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          # print(newImage.shape)
          #print(newImage[1][4][0]* 255)
          
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
               newImage.save(f"{path}{sj}{action}{item_idx}gasf.png")
          # plt.close('all')
     else:
          newImage = RGBfromMTFMatrix_of_XYZ((_r+1)/2, (_g+1)/2,(_b+1)/2)
          newImage = Image.fromarray((newImage * 255).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}gasf.png")
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
def Reconstruct_GAF(img,dictionary,a):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    
    _r=np.interp(_r,(0,255),(-1,1))
    _g=np.interp(_g,(0,255),(-1,1))
    _b=np.interp(_b,(0,255),(-1,1))
    
    r=GAF_to_TS(_r,0,dictionary[a])
    g=GAF_to_TS(_g,1,dictionary[a])
    b=GAF_to_TS(_b,2,dictionary[a])
    N=[]
    N.append(r)
    N.append(g)
    N.append(b)
    return N
def GAF_to_TS(gaf,i,dictionary,method='summation'):
    #We recontruct first phi vector based on gasf matrix diagonal property
    print(gaf)
    phi=np.arccos(np.diag(gaf))/2
    
    X_normalized=np.cos(phi)
    
    
    #RECONSTRUCCIÓN DISCRETA DE LOS ESTADOS:
    #Sabemos que el valor medio de los datos maximos es 
    MAX=dictionary[0] 
    MIN=dictionary[1]
    

    #x=np.interp(X_normalized,(np.min(X_normalized),np.max(X_normalized)),(MIN[i],MAX[i]))
    x=X_normalized
       
     

    # Transformar los puntos
     #posi =min_b + n * (max_b - min_b)
    x=np.interp(x,(np.min(x),np.max(x)),(MIN[i], MAX[i])) 

    return x
def generate_and_save_grammian_angular_field(fold, dataset_folder, training_data, y_data, sj_train, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
    subject_samples = 0
    p_bar = tqdm(range(len(training_data)))

    for i in p_bar:
      w = training_data[i]
      sj = sj_train[i][0]
      w_y = y_data[i]
      w_y_no_cat = np.argmax(w_y)
      print("w_y", w_y, "w_y_no_cat", w_y_no_cat)
      print("w", w.shape)

      # Update Progress Bar after a while
      time.sleep(0.01)
      p_bar.set_description(f'[{data_type} | FOLD {fold} | Class {w_y_no_cat}] Subject {sj}')
    
      # #-------------------------------------------------------------------------
      # # only for degugging in notebook
      # #-------------------------------------------------------------------------
      # df_original_plot = pd.DataFrame(w, columns=["x_axis", "y_axis", "z_axis"])
      # df_original_plot["signal"] = np.repeat("Original", df_original_plot.shape[0])
      # df_original_plot = df_original_plot.iloc[:-1,:]
      # plot_reconstruct_time_series(df_original_plot, "Walking", subject=sj)
      # #-------------------------------------------------------------------------

      #print(f"{'*'*20}\nSubject: {sj} (window: {i+1}/{len(training_data)} | label={y})\n{'*'*20}")
      #print("Window shape",w.shape)
      if fold < 0:
        img = SavevarGAF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/GAF/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
      else:
        img = SavevarGAF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
      print("w image (RP) shape:", np.array(img).shape)
      
      
      subject_samples += 1
def generate_all_grammian_angular_field(X_train, y_train, sj_train, dataset_folder="/home/adriano/Escritorio/TFG/data/WISDM/", TIME_STEPS=129,  FOLDS_N=3, sampling="loto"):
  groups = sj_train 
  if sampling == "loto":
    #TODO change 100 for an automatic extracted number greater than the max subject ID: max(sj_train)*10
    groups = [[int(sj[0])+i*100+np.argmax(y_train[i])+1] for i,sj in enumerate(sj_train)]

  # if DATASET_NAME == "WISDM": #since wisdm is quite balanced
  sgkf = StratifiedGroupKFold(n_splits=FOLDS_N)
  # elif DATASET_NAME == "MINDER" or DATASET_NAME == "ORIGINAL_WISDM": 
  #   sgkf = StratifiedGroupKFold(n_splits=FOLDS_N)

  accs = []
  y_train_no_cat = [np.argmax(y) for y in y_train]
  p_bar_classes = tqdm(range(len(np.unique(y_train_no_cat))))
  all_classes = np.unique(y_train_no_cat)
  print("Classes available: ", all_classes)
  for fold in range(FOLDS_N):
    for i in p_bar_classes:
        y = all_classes[i]
        time.sleep(0.01) # Update Progress Bar after a while
        os.makedirs(f"{dataset_folder}plots/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True)
        os.makedirs(f"{dataset_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True)

  for fold, (train_index, val_index) in enumerate(sgkf.split(X_train, y_train_no_cat, groups=groups)):

    # if fold != 2:
    #   continue

    # print(f"{'*'*20}\nFold: {fold}\n{'*'*20}")
    # print("Train index", train_index)
    # print("Validation index", val_index)
    training_data = X_train[train_index,:,:]
    validation_data = X_train[val_index,:,:]
    y_training_data = y_train[train_index]
    y_validation_data = y_train[val_index]
    sj_training_data = sj_train[train_index]
    sj_validation_data = sj_train[val_index]

    print("training_data.shape", training_data.shape, "y_training_data.shape", y_training_data.shape, "sj_training_data.shape", sj_training_data.shape)
    print("validation_data.shape", validation_data.shape, "y_validation_data.shape", y_validation_data.shape, "sj_validation_data.shape", sj_validation_data.shape)



    generate_and_save_grammian_angular_field(fold, dataset_folder, training_data, y_training_data, sj_training_data, TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    generate_and_save_grammian_angular_field(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    

def main():
     data_name="WISDM"
     data_folder="/home/adriano/Escritorio/TFG/data/WISDM/"
     #voy a  obtener el maximo de todo el data set.
     X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
     print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
     #print(sj_train[:,0])
     #print(y_train[:,0])
     #print(X_train)
     #print("minimo",np.min(X_train))
     #print("maximo",np.max(X_train))
     MAX=np.max(X_train)
     MIN=np.min(X_train)
     a=20
     w = X_train[a]
     sj = sj_train[a][0]
     w_y = y_train[a]
     w_y_no_cat = np.argmax(w_y)
     print(w.shape)
     """
     valoresa=np.linspace(-5, 20, 129)
     valoresb=np.linspace(-20, 50, 129)
     valoresc=np.linspace(-10, 3, 129)
     experimento=np.array([valoresa,valoresb,valoresc]).reshape(129,3)
     experimentoinv=np.array([valoresa[::-1],valoresb[::-1],valoresc[::-1]]).reshape(129,3)
     
     img = SavevarGAF_XYZ(experimento, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
     img2 = SavevarGAF_XYZ(experimentoinv, sj, 1, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     """
     dictionary=dict()
     for k in range(0,8163):
         l=X_train[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]
     #w=w*-1
     #w[:, 0]=w[:, 0]*-1
     img = SavevarGAF_XYZ(w, sj, a, "x", normalized = 1, path=f"./", TIME_STEPS=129)
     imagen = cv2.imread("./1600x20gasf.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     
     rp=Reconstruct_GAF(imagen,dictionary,a)
     # Configurar el estilo de los gráficos
    
     dim=0
     
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
     error_absoluto, error_relativo = calcular_errores(f, rp[dim])
     #d = dtw.distance_fast(f, rp[1], use_pruning=True)
     print(f"Error Absoluto Promedio: {error_absoluto}")
     print(f"Error Relativo Promedio: {error_relativo}")
     #print(f"Error DTW: {d}")
     print(f"Coeficiente de correlación: {np.corrcoef(f, rp[dim])[0,1]}")
     

if __name__ == '__main__':
    main()
