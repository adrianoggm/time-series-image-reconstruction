#!/home/adriano/Escritorio/TFG/venv/bin/python3
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
#import activity_data as act
import tgen.activity_data as act
#import recurrence_plots as rec
import tgen.recurrence_plots as rec
from pyts.image import MarkovTransitionField
from PIL import Image
import cv2
from sklearn.manifold import MDS
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import os
import time
import seaborn as sns
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
    num_states = TIME_STEPS
    mtf = MarkovTransitionField(image_size=num_states,overlapping=True,n_bins=4)
    
    X_mtf = mtf.fit_transform(x).reshape(num_states, num_states)
    
    return X_mtf

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
    num_states = 4
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

def generate_and_save_markov_transition_field(fold, dataset_folder, training_data, y_data, sj_train, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
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
        img = SavevarMTF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/MTF/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
      else:
        img = SavevarMTF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
      print("w image (RP) shape:", np.array(img).shape)
      
      
      subject_samples += 1
      
     
def SavevarMTF_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
    if not all([(x==0).all()]):
     _r = varMTF2(x,'x', TIME_STEPS)
     _g = varMTF2(x,'y', TIME_STEPS)
     _b = varMTF2(x,'z', TIME_STEPS)

     #print("X", _r[0])
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
def Reconstruct_MTF(img,dictionary,valor):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    
    _r=np.interp(_r,(0,255),(0,1))
    _g=np.interp(_g,(0,255),(0,1))
    _b=np.interp(_b,(0,255),(0,1))
    #print(_r)
    r=MTF_to_TS(_r,dictionary[valor],0)
    g=MTF_to_TS(_g,dictionary[valor],1)
    b=MTF_to_TS(_b,dictionary[valor],2)
    N=[]
    N.append(r)
    N.append(g)
    N.append(b)
    return N
def cosine_similarity(vec1, vec2):
        """
        Calcula la similitud del coseno entre dos vectores.
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)

def MTF_to_TS(mtf,dictionary,index=0,numstates=4,TIMESTEPS=129):
    similaritymatrix=np.zeros_like(mtf)
    #CLASIFICACIÓN 
    #Cálculo matriz de similaridades
    for i in range(0,TIMESTEPS):
        for j in range(0,TIMESTEPS):
            if i<=j:
                similaritymatrix[i][j]=cosine_similarity(mtf[i,:],mtf[j,:])
                similaritymatrix[j][i]=similaritymatrix[i][j]
    #Clasificación de los estados
    estados=[]
    estado=[0]
    estadosfaltantes=np.arange(0, 129)
    nvalores=TIMESTEPS//numstates
    #print(len(estadosfaltantes),nvalores)
    for i in range(0,numstates):
        estado=[estadosfaltantes[0]]
        estadosfaltantes=np.delete(estadosfaltantes, 0)
        
        for j in range(0,nvalores-1):
            v=[]
            #print("LLEGA",j)
            
            for k in range(0,len(estadosfaltantes)):
                set2=[]
                set2.append(estadosfaltantes[k])
                set1=estado
                similarities=[similaritymatrix[vec1][vec2]for vec1, vec2 in zip(set1, set2)]
                v.append(np.mean(similarities))
            newval=np.argmax(v)
            estado.append(estadosfaltantes[newval])
            estadosfaltantes=np.delete(estadosfaltantes, newval)
            #print("TAMANO",len(estado))
        estados.append(estado)
           
    
    #lo meto al ultimo estado 
    estados[0].append(estadosfaltantes[0])
    #print("ESTADOS",estados)
    statematrix=np.zeros((numstates, numstates))
    for i in range(0,numstates):
        for j in range(0,numstates):
            if j!=i and i<=j:
                set1=estados[i] 
                set2= estados[j]   
                similarities=[similaritymatrix[vec1][vec2]for vec1, vec2 in zip(set1, set2)]
                statematrix[i][j]=np.mean(similarities)
                statematrix[j][i]=np.mean(similarities)
    # Calcular la similitud media para cada valor del vector
    #print(statematrix)
    
    vector=np.arange(0, numstates) #(0,1,2,3)
    filasactuales=np.arange(0, numstates)
    indexval=numstates//2
    tr=np.arange(0, numstates)
    #We take as pivot the maxsimilarity value
    mean_similarities = np.sum(statematrix, axis=1)
    pivote=np.argmax(mean_similarities)
    aux=vector[indexval]    
    vector[indexval]=pivote
    vector[pivote]=aux
    #We arange the vector.
    #caso par 
    izq_indx=numstates//2-1 #1 2
    izq_val=pivote
    dch_indx=numstates//2+1 #3 
    dch_val=pivote
    #print(pivote,statematrix)
    m=statematrix
    count=1 # ya hay un estado colocado
    while count!=numstates:
        a=[]
        for i in range(0,len(m)):
            if(i!=np.argmax(mean_similarities)):
                a.append(m[i,:])
        m=np.array(a)
        #print(izq_val,m)
        #print("TR",tr)
        tr= np.delete(tr,np.where(tr==dch_val))
        #print("TR",tr)
        #tr contiene los indices correspondientes a la matriz m
        mean_similarities = np.sum(m, axis=1)
        izq_val=tr[np.argmax(mean_similarities)]
        vector[izq_indx]=izq_val
        izq_indx-=1        
        count+=1
        if count==numstates:
            break
        tr= np.delete(tr,np.where(tr==izq_val))
        a=[]
        for i in range(0,len(m)):
            if(i!=np.argmax(mean_similarities)):
                a.append(m[i,:])
        
        m=np.array(a)
        mean_similarities = np.sum(m, axis=1)
        dch_val=tr[np.argmax(mean_similarities)]
        vector[dch_indx]=dch_val
        dch_indx-=1        
        count+=1
       


    
    valores=vector
    #print(valores)
    f=np.array([-1,1,3,5])
    serie=np.arange(0, 129)
    #print(estados)
    for i in range(0,numstates):
        for j in range(0,len(estados[i])):
            #print(valores[i])
            est=f[valores[i]]
            serie[estados[i][j]]=est
            
    #print(serie)
    min_a, max_a = np.min(serie), np.max(serie)  
     
     #nega = min_b + n * (max_b - min_b)
    MAX=dictionary[0]
    MIN=dictionary[1]
    
    serie=np.interp(serie,(min_a,max_a),(MIN[index], MAX[index]))
    return serie
def generate_all_markov_transition_field(X_train, y_train, sj_train, dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", TIME_STEPS=129,  FOLDS_N=3, sampling="loso"):
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
        os.makedirs(f"{dataset_folder}plots/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
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



    generate_and_save_markov_transition_field(fold, dataset_folder, training_data, y_training_data, sj_training_data, TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    generate_and_save_markov_transition_field(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    

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
     a=0
     w = X_train[a]
     sj = sj_train[a][0]
     w_y = y_train[a]
     w_y_no_cat = np.argmax(w_y)
     
     print(w.shape)
     dictionary=dict()
     for k in range(0,8163):
         l=X_train[k]
         #A=np.max(l[:,0])
         maximos=[np.max(l[:,0]),np.max(l[:,1]),np.max(l[:,2])]
         minimos=[np.min(l[:,0]),np.min(l[:,1]),np.min(l[:,2])]
         dictionary[k]=[maximos,minimos]
     img = SavevarMTF_XYZ(w, sj, 0, "x", normalized = 1, path=f"./", TIME_STEPS=129) 
     
     
     
     imagen = cv2.imread("./1600x0mtf.png")  
     imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
     print("Image shape",imagen.shape)
     rp=Reconstruct_MTF(imagen,dictionary,a)
     # Configurar el estilo de los gráficos
     
     dim=1
     
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
     #plt.legend(loc="upper left",bbox_to_anchor=(1,1))
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