import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import math
# from tqdm.notebook import tqdm, trange #progress bars for jupyter notebook
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)

import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from pyts.image import MarkovTransitionField
from PIL import Image

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
               newImage.save(f"{path}{sj}{action}{item_idx}.png")
          # plt.close('all')
     else:
          newImage = RGBfromMTFMatrix_of_XYZ(_r, _g, _b)
          newImage = Image.fromarray((newImage * 255).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}.png")
          # plt.close('all')
     return newImage
    else:
     return None
  


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
    


    