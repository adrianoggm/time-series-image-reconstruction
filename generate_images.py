#!/home/adriano/Escritorio/TFG/venv/bin/python3
from keras.utils import to_categorical
import argparse
import tgen.activity_data as act
import tgen.recurrence_plots as rec
import tgen.GAF as gaf
import tgen.MTF as mtf
import numpy as np
import os
from tqdm.auto import trange, tqdm
import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

def generate_and_save_images(fold, dataset_folder, training_data, y_data, sj_train, TIME_STEPS=129, data_type="train", single_axis=False, FOLDS_N=3, sampling="loso",image_type="RP"):
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
      if image_type=="RP":
        if fold < 0:
            img = rec.SavevarRP_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
        else:
            img = rec.SavevarRP_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
        print("w image (RP) shape:", np.array(img).shape)
        
        if single_axis and img is not None:
            #also, save each single data column values
            for col in range(w.shape[1]):
                if fold < 0:
                    img = rec.SavevarRP_fran(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
                else:
                    img = rec.SavevarRP_fran(w, col, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/single_axis/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
                print("w single axis (RP) shape:", img.shape)
      if image_type=="MTF":
            if fold < 0:
                img = mtf.SavevarMTF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/MTF/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
            else:
                img = mtf.SavevarMTF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/MTF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)   
      if image_type=="GAF":
            if fold < 0:
                img = gaf.SavevarGAF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/GAF/sampling_{sampling}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
            else:
                img = gaf.SavevarGAF_XYZ(w, sj, subject_samples, "x", normalized = 1, path=f"{dataset_folder}plots/GAF/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/{data_type}/{w_y_no_cat}/", TIME_STEPS=TIME_STEPS)
            
      subject_samples += 1

def generate_all_images(X_train, y_train, sj_train, dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", TIME_STEPS=129,  FOLDS_N=3, sampling="loso",imagetype="RP"):
  name=imagetype
  if(name=="RP"):
     name="recurrence_plot"
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
        os.makedirs(f"{dataset_folder}plots/{name}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/{y}/", exist_ok=True) 
        os.makedirs(f"{dataset_folder}plots/{name}/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/test/{y}/", exist_ok=True) 
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

    #arch_training_data=f"{dataset_folder}tseries/recurrence_plot/sampling_{sampling}/{FOLDS_N}-fold/fold-{fold}/train/training_data.npy"
    #np.save(arch_training_data,np.array(train_index))
   
    generate_and_save_images(fold, dataset_folder, training_data, y_training_data, sj_training_data, TIME_STEPS=TIME_STEPS, data_type="train", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling,image_type=imagetype)
    generate_and_save_images(fold, dataset_folder, validation_data, y_validation_data, sj_validation_data, TIME_STEPS=TIME_STEPS, data_type="test", single_axis=False, FOLDS_N=FOLDS_N, sampling=sampling)
    
def main():
    '''Examples of runs:
    $ nohup ./generate_images.py  --data-name WISDM --n-folds 3 --image-type GAF --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loto > recurrence_plots_loto.log &
    - load LOSO numpies
    nohup ./generate_recurrence_plots.py --data-name WISDM --n-folds 3 --data-folder /home/adriano/Escritorio/TFG/data/WISDM/ --sampling loso  > recurrence_plots_loso.log &
    
    
    $ nohup ./generate_recurrence_plots.py --data-name WISDM --n-folds 3 --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loso  > recurrence_plots_loso.log &
    
    - Create numpies included
    $ nohup ./generate_recurrence_plots.py --create-numpies --data-name WISDM --n-folds 3 --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loso > recurrence_plots_loto.log &
    $ nohup ./generate_recurrence_plots.py --create-numpies --data-name WISDM --n-folds 3 --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loto > recurrence_plots_loto.log &
    nohup ./generate_recurrence_plots.py --data-name WISDM --n-folds 3 --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loso  > recurrence_plots_loso.log &
    
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--data-name', type=str, default="WISDM", help='the database name')
    p.add_argument('--data-folder', type=str, default="/home/adriano/Escritorio/TFG/data/WISDM/", help='the data folder path')
    p.add_argument('--n-folds', type=int, default=3, help='the number of k-folds')
    p.add_argument('--sampling', type=str, default="loso", help='loso: leave-one-subject-out; loto: leave-one-trial-out')
    p.add_argument('--create-numpies', action="store_true", help='create numpies before; if not, load numpies')
    p.add_argument('--image-type', type=str, default="RP", help='they can be RP,GAF,MTF')

    args = p.parse_args()
    create_numpies = args.create_numpies
    data_folder = args.data_folder
    data_name = args.data_name
    img_type=args.image_type
    FOLDS_N = args.n_folds
    TIME_STEPS, STEPS = act.get_time_setup(DATASET_NAME=data_name)
    print("TIME_STEPS", TIME_STEPS, "STEPS", STEPS)
    X_train, y_train, sj_train = None, None, None
    if not create_numpies:
        print("Loading numpies...")
        X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
    else:
        print("Creating numpies...")
        X_train, y_train, sj_train = act.create_all_numpy_datasets(data_name, data_folder, COL_SELECTED_IDXS=list(range(3, 3+3)))
        y_train = to_categorical(y_train, dtype='uint8') 
    print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
    
    generate_all_images(X_train, y_train, sj_train, data_folder, TIME_STEPS, FOLDS_N, args.sampling,img_type)


if __name__ == '__main__':
    main()