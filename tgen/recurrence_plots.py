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

from PIL import Image

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
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)

        newImage = Image.fromarray((newImage * 255).astype(np.uint8))

        # plt.imshow(newImage)
        if saveImage:
          # plt.savefig(f"{path}{sj}{action}{item_idx}_rp.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
          newImage.save(f"{path}{sj}{action}{item_idx}_rp.png")
        # plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))

        # plt.imshow(newImage)
        if saveImage:
          # plt.savefig(f"{path}{sj}{action}{item_idx}_rp.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
          newImage.save(f"{path}{sj}{action}{item_idx}_rp.png")

        # plt.close('all')
    return newImage
def SavevarRP_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=129):
  if not all([(x==0).all()]):
    _r = varRP(x,'x', TIME_STEPS)
    _g = varRP(x,'y', TIME_STEPS)
    _b = varRP(x,'z', TIME_STEPS)

    #print("X", _r)
    #print("Y", _g)
    #print("Z", _b)

    # plt.close('all')
    # plt.figure(figsize=(1,1))
    # plt.axis('off')
    # plt.margins(0,0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())

    print("fig size: width=", plt.figure().get_figwidth(), "height=", plt.figure().get_figheight())

    if normalized:
        newImage = RGBfromRPMatrix_of_XYZ(NormalizeMatrix(_r), NormalizeMatrix(_g), NormalizeMatrix(_b))
        #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        # print(newImage.shape)
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))
        # plt.imshow(newImage)
        if saveImage:
          # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
          newImage.save(f"{path}{sj}{action}{item_idx}.png")
        # plt.close('all')
    else:
        newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
        newImage = Image.fromarray((newImage * 255).astype(np.uint8))
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

# def recover_time_serie_from_varRP(varRP):
#     dimR = len(varRP)

#     #R = np.zeros((dimR,dimR))
#     R = np.eye(dimR)
#     for i in range(dimR):
#         for j in range(dimR):
#             if Cosin2vec(list(map(lambda x:x[0]-x[1], zip(s[i], s[j]))), [1,1]) >= pow(2, 0.5)/2:
#                 sign =1.0
#             else:
#                 sign =-1.0
#             R[i][j] = sign*Distance2dim(s[i],s[j])
#     return R

#@title plot  reconstruction

def plot_reconstruct_time_series( df, activity, dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", TIME_STEPS=129, axes_cols=['x_axis', 'y_axis', 'z_axis'], signal_col="signal", timestep_start_idx=0, subject=0, grid=True, figsize=(10,4), saveFig=False, showPlot=False):
  sns.set_style("darkgrid")

  columns = axes_cols.copy()
  columns.append(signal_col)
  data = df[columns]
  # print(data)
  
  if len(data) > 0:
    # print("Plots")
    # fig, ax = plt.subplots(3,1)
    # for ax in ax.flatten():
    #   data.set_index('signal',append=True).unstack().plot(ax=ax,subplots=True, figsize=figsize, title=activity, xticks=range(0, TIME_STEPS+1, TIME_STEPS // 2))
    df_list = []
    for y in axes_cols:
      # print(y)
      # print( len(axes_cols) >1)
      d = data.set_index(signal_col,append=True).unstack()[y] #if len(axes_cols) >1 else data
      df_list.append(d)
      # print(d)
      # axis = d.plot(figsize=figsize, title=activity, xticks=range(0, TIME_STEPS+1, TIME_STEPS // 2))
      # for ax in axis:
      # axis.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
      # axis.grid(grid)

    # make a list of all dataframes 
    nrows=len(axes_cols)

    fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True) #, sharey=True)
    # print("axes.shape", axes.shape)
    # plot counter
    count=0
    for r in range(nrows):

      df_list[count].plot(ax=axes[r,], figsize=figsize, xticks=range(0, len(df) // 2 +1, (len(df) // 2) // 4))
      axes[r,].legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
      axes[r,].grid(grid)
      axes[r,].set_ylabel(axes_cols[r])
      count+=1
    
    plt.xlabel("Timestep")
    fig.suptitle(activity)


    # axis = data.plot(subplots=True, figsize=figsize, title=activity, xticks=range(0, TIME_STEPS+1, TIME_STEPS // 2))
    # for ax in axis:
    #     ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))
    #     ax.grid(grid)

    save_path = f"{dataset_folder}visualization/{activity}/time-series-reconstruction/{subject}/"
    os.makedirs(save_path, exist_ok=True)
    # fig = axis.get_figure()
    # plt.tight_layout()
    
    if saveFig:
      # print(f"{save_path}visual_cl_{activity}_timestep_{TIME_STEPS}_w_{timestep_start_idx}.png")
      fig.savefig(f"{save_path}visual_cl_{activity}_timestep_{TIME_STEPS}_w_{timestep_start_idx}.png", dpi=300, bbox_inches='tight')

    if showPlot:
      plt.show()
    plt.close('all')

  else:
    print("no data to plot")

def generate_and_save_recurrence_plot(fold, dataset_folder, training_data, y_data, sj_train, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso"):
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
        img = SavevarRP_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
      else:
        img = SavevarRP_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
      print("w image (RP) shape:", np.array(img).shape)
      
      if single_axis and img is not None:
        #also, save each single data column values
        for col in range(w.shape[1]):
          if fold < 0:
            img = SavevarRP_fran(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
          else:
            img = SavevarRP_fran(w, col, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
          print("w single axis (RP) shape:", img.shape)
      subject_samples += 1
      
     


def generate_all_recurrence_plots(X_train, y_train, sj_train, dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", TIME_STEPS=129,  FOLDS_N=3, sampling="loso"):
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
        os.makedirs(f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
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



    generate_and_save_recurrence_plot(fold, dataset_folder, training_data, y_training_data, sj_training_data, TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    generate_and_save_recurrence_plot(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    


    