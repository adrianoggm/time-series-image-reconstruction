
import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical


# def download_data():
#     if DATASET_NAME == "ADL_Dataset":
#         !gdown https://archive.ics.uci.edu/ml/machine-learning-databases/00283/ADL_Dataset.zip  &>> instal.log
#         !python -m zipfile -e ADL_Dataset.zip ADL_Dataset/  &>> instal.log
#     elif DATASET_NAME == "ORIGINAL_WISDM":
#     # else:
#     !pip install gdown &> install.log
#     print("downloading ORIGINAL_WISDM")
#     !gdown --id 152sWECukjvLerrVG2NUO8gtMFg83RKCF --output WISDM_ar_latest.tar.gz
#     if USE_REMOTE_SERVER:
#         !tar -xvf WISDM_ar_latest.tar.gz -C data/ORIGINAL_WISDM/
#     else:
#         !tar -xvf WISDM_ar_latest.tar.gz -C /content/drive/MyDrive/Databases/diffusion/ORIGINAL_WISDM/

def get_time_setup(DATASET_NAME = "WISDM"):
    #In WISDM, since sampling rate is 20Hz: 20Hz * seconds of duration
    if DATASET_NAME == "WISDM" or DATASET_NAME == "ORIGINAL_WISDM":
        DATASET_HZ = 20 #WISDM 20Hz; ADL_Dataset 32Hz
        TIME_SECS = 6.4#2.56 #in secs
    elif DATASET_NAME == "ADL_Dataset": 
        DATASET_HZ = 32
        TIME_SECS = 10 #in secs
    elif "MINDER":
        DATASET_HZ = 4 #4 hours
        TIME_SECS = 60 #in secs
    else:
        DATASET_HZ = 32

    TIME_STEPS = int(DATASET_HZ*TIME_SECS) +1 #+1 por los recurrence plots
    STEP = TIME_STEPS  #without overlapping; 50% ovelaps == TIME_STEPS//2
    print("datset: ", DATASET_NAME, "Sampling rate:", DATASET_HZ, "samples per window:", TIME_STEPS)
    return TIME_STEPS, STEP


def create_wisdm_dataset(path):
    activities = {"A": "Walking", "B": "Jogging", "C": "Stairs", "D": "Sitting",
                  "E": "Standing","F": "Typing", "G": "Brushing Teeth", "H":"Eating Soup",
                  "I": "Eating Chips", "J": "Eating Pasta", "K": "Drinking from Cup",
                  "L": "Eating Sandwich", "M": "Kicking (Soccer Ball)", "O": "Playing Catch w/Tennis Ball",
                  "P": "Dribbling (Basketball)", "Q": "Writting", "R": "Clapping", "S": "Folding Clothes"} 
    df = pd.DataFrame()
    for f in sorted(os.listdir(path)):
        daux = pd.read_csv(path+f, header=None, names=['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis'])
        daux.activity = daux.activity.apply(lambda x: activities[x])
        daux.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
        daux['z_axis'] = daux.z_axis.astype(np.float64)
        daux.dropna(axis=0, how='any', inplace=True)
        df = pd.concat([df, daux])
    return df

def preprocess_activity_data_sets(DATASET_NAME="WISDM", dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", column_names = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']):

    if DATASET_NAME == "MINDER":
        df = pd.read_csv(f"{dataset_folder}minder-preprocessed.csv")
        df = df.rename(columns={"uti_label": "activity", "patient_id": "user_id", #  "freq|Bathroom": "x_axis", "freq|Bed_in": "y_axis", "freq|Bed_out": "z_axis"
                        })

    elif DATASET_NAME == "WISDM":
        if os.path.exists(f"{dataset_folder}wisdm-watch-acc-dataset.csv"):
            df = pd.read_csv(f"{dataset_folder}wisdm-watch-acc-dataset.csv")
        else:
            data_path = f"{dataset_folder}wisdm-dataset/raw/"
            phone_path = f"{data_path}phone/accel/"
            watch_path = f"{data_path}watch/accel/"            
            watch_df = create_wisdm_dataset(watch_path)
            phone_df = create_wisdm_dataset(phone_path)
            df = pd.concat([watch_df, phone_df])
            df.to_csv(f"{dataset_folder}wisdm-watch-and-phone-dataset.csv", index=None)
            watch_df.to_csv(f"{dataset_folder}wisdm-watch-acc-dataset.csv", index=None)
            phone_df.to_csv(f"{dataset_folder}wisdm-phone-acc-dataset.csv", index=None)
            df = watch_df.copy()
        df = df[df.activity.isin(['Walking', 'Jogging', 'Stairs', 'Sitting', 'Standing'])]
        if os.path.exists(f"{dataset_folder}wisdm-watch-acc-dataset.csv"):
            watch_df = pd.read_csv(f"{dataset_folder}wisdm-watch-acc-dataset.csv")
        if os.path.exists(f"{dataset_folder}wisdm-phone-acc-dataset.csv"):
            phone_df = pd.read_csv(f"{dataset_folder}wisdm-phone-acc-dataset.csv")
    elif DATASET_NAME == "ORIGINAL_WISDM":
        # class_names = ['Downstairs', 'Jogging', 'Sitting', 'Standing', 'Upstairs', 'Walking']
        df = pd.read_csv(f'{dataset_folder}WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header=None, names=column_names)
        df.z_axis.replace(regex=True, inplace=True, to_replace=r';', value=r'')
        df['z_axis'] = df.z_axis.astype(np.float64)
        df.dropna(axis=0, how='any', inplace=True)
    elif DATASET_NAME == "ADL_Dataset":
        #SELECTION OF CLASSES BASED ON RESULTS OF ORIGINAL PAPER: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6630784
        #Although in Recurrece Plot paper used other classes: https://s3.us-west-2.amazonaws.com/secure.notion-static.com/49ad9327-4337-4815-8472-15e77ed153b0/Robust_Single_Accelerometer-Based_Activity_Recognition_Using_Modified_Recurrence_Plot.pdf?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAT73L2G45EIPT3X45%2F20220922%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20220922T153910Z&X-Amz-Expires=86400&X-Amz-Signature=15312ceb0ee9c5d01ee994c7cce25e4bcbf3a85821e52902aaad36bd910f499e&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22Robust_Single_Accelerometer-Based_Activity_Recognition_Using_Modified_Recurrence_Plot.pdf%22&x-id=GetObject
        #Original ddbb paper got better results with the following classes:
        adls = ["Climb_stairs", "Drink_glass", "Getup_bed", "Pour_water", "Sitdown_chair", "Standup_chair", "Walk"]
        # adls = ["Brush_teeth", "Climb_stairs", "Comb_hair", "Descend_stairs"
        # , "Drink_glass", "Eat_meat", "Eat_soup", "Getup_bed", "Liedown_bed"
        # , "Pour_water", "Sitdown_chair", "Standup_chair", "Use_telephone", "Walk"]

        # class_names = adls
        dfs = []

        activity_labels = []
        final_ids = []
        user_ids = []
        df_final = []
        timestamps = []
        for adl in adls:
            path = f'ADL_Dataset/HMP_Dataset/{adl}/'
            print(f"Processing {adl}")
            files = os.listdir(path)
            for f in files:
                id = f.split("-")[8].split(".")[0]
                if id not in final_ids:
                    final_ids.append(id)
                # print("Subject:", id)
                df = pd.read_csv(path+f, sep=" ", header=None)
                dfs.append(df)
                timestamps.extend(list(range(len(df))))
                user_ids.extend(np.repeat(final_ids.index(id), len(df)))
                activity_labels.extend(np.repeat(adl, len(df)))
        df = pd.concat(dfs, ignore_index=True)
        df["user_id"] = user_ids
        df["activity"] = activity_labels
        df["timestamp"] = timestamps

        df = df.rename(columns={0: "x_axis", 1: "y_axis", 2: "z_axis"})
        df = df[column_names]
        df.to_csv("res.csv")
    
    classes = list(df.activity.unique())
    df["activity"] = df["activity"].apply(lambda x: classes.index(x))

    print(" users:", np.unique(df['user_id']), "\ntotal:", len(np.unique(df['user_id'])))

    return df


from scipy import stats
import os    
import shutil

def create_dataset(savepath, data_type, X, y, sj_id, time_steps=1, step=1, verbose=False):
    Xs_all, ys_all, sjs_all = [], [], []
    Xs, ys, sjs = [], [], []
    l_prev = stats.mode(y.iloc[: time_steps])[0][0]
    if type(l_prev) == str:
          l_prev = l_prev.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")#.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")
    # print("l prev:", l_prev)
    sj_prev = stats.mode(sj_id.iloc[: time_steps])[0][0]

    #delete first all previous data
    shutil.rmtree(f"{savepath}{data_type}/windowed/", ignore_errors=True)

    for i in range(0, len(X) - time_steps, step):
      v = X.iloc[i:(i + time_steps)].values #a window
      if not all([(v==0).all()]):
        labels = y.iloc[i: i + time_steps]
        sj_ids = sj_id.iloc[i: i + time_steps]
        l = stats.mode(labels)[0][0]
        if type(l) == str:
          l = l.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")
        sj = stats.mode(sj_ids)[0][0]
        # print("l current:", l)
        # print("subject:", sj)
        # print("index:", i + time_steps)

        #-------------------------------------------------------------------------
        # only for testing plot every window
        #-------------------------------------------------------------------------
        if verbose:
          print(f"Ploting window raw --- LABEL [{l}] . SUBJECT [{sj}] | Window id [{i}] | win shape: {v.shape}")
          df_original_plot = pd.DataFrame(v, columns=["x_axis", "y_axis", "z_axis"])
          # print(df_original_plot)
          df_original_plot["signal"] = np.repeat("Original", df_original_plot.shape[0])
          df_original_plot = df_original_plot.iloc[:-1,:]
          print(df_original_plot)
        #   plot_reconstruct_time_series(df_original_plot, l, subject=sj_prev)
        #-------------------------------------------------------------------------
        
        if l != l_prev:
            # print("l prev:", l_prev)

            # print("l current:", l)

            print(f"Saving x data [subject {sj_prev}], label: {l_prev}, size: {np.array(Xs).shape}")
            # os.makedirs(f"{savepath}{data_type}/RP/{l_prev}/", exist_ok=True)
            # np.save(f"{savepath}{data_type}/RP/{l_prev}/x.npy", np.array(Xs))
            

            if os.path.exists(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy"):
              print("Exists:", f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy. Loading existing file and appending the new data")
              aux = np.load(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy")
              for a in aux:
                Xs.append(a)
            #-------------------------------------------------------------------------
            # only for testing plot every window
            #-------------------------------------------------------------------------
            if verbose:
              for wi,w in enumerate(np.array(Xs)):
                print(f"Ploting --- LABEL [{l_prev}] . SUBJECT [{sj}] | Window id [{wi}]")
                df_original_plot = pd.DataFrame(w, columns=["x_axis", "y_axis", "z_axis"])
                # print(df_original_plot)
                df_original_plot["signal"] = np.repeat("Original", df_original_plot.shape[0])
                df_original_plot = df_original_plot.iloc[:-1,:]
                print(df_original_plot)
                # plot_reconstruct_time_series(df_original_plot, "Walking", subject=sj_prev)
            #-------------------------------------------------------------------------

              
            os.makedirs(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/", exist_ok=True)
            np.save(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy", np.array(Xs))

            #np.save(f"{savepath}{data_type}/windowed/{l_prev}/{sj}/y.npy",  np.array(ys).reshape(-1, 1))
            #np.save(f"{savepath}{data_type}/windowed/{l_prev}/{sj}/sj.npy", np.array(sjs).reshape(-1, 1))
            Xs, ys, sjs = [], [], []
            l_prev = l
        if sj != sj_prev:
          sj_prev = sj

        Xs.append(v)        
        ys.append(l)
        sjs.append(sj)
        Xs_all.append(v)
        ys_all.append(l)
        sjs_all.append(sj)
 

    # print("sj:", sj, "seg:",i, "/", len(X) - time_steps, "label:", l)
    
    print(f"LAST -- >Saving x data [subject {sj_prev}], label: {l_prev}, size: {np.array(Xs).shape}")
    os.makedirs(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/", exist_ok=True)

    if os.path.exists(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy"):
      print("Exists:", f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy")
      aux = np.load(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy")
      for a in aux:
        Xs.append(a)
            
    np.save(f"{savepath}{data_type}/windowed/{l_prev}/{sj_prev}/x.npy", np.array(Xs))

    np.save(f"{savepath}{data_type}/windowed/X_{data_type}.npy", np.array(Xs_all))
    np.save(f"{savepath}{data_type}/windowed/y_{data_type}.npy", np.array(ys_all).reshape(-1, 1))
    np.save(f"{savepath}{data_type}/windowed/sj_{data_type}.npy", np.array(sjs_all).reshape(-1, 1))

    
    return np.array(Xs_all), np.array(ys_all).reshape(-1, 1), np.array(sjs_all).reshape(-1, 1)


def create_all_numpy_datasets(DATASET_NAME="WISDM", dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", COL_SELECTED_IDXS=list(range(3, 3+3))):
    TIME_STEPS, STEP = get_time_setup(DATASET_NAME)
    df = preprocess_activity_data_sets(DATASET_NAME, dataset_folder, column_names = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis'])
    print("Classes preprocessed:", np.unique(df.activity))
    # create train dataset of numpies
    X_train, y_train, sj_train = create_dataset(
        dataset_folder+"numpies/",
        "train",
        df.iloc[:, COL_SELECTED_IDXS], 
        df.activity,
        df.user_id ,
        TIME_STEPS, 
        STEP,
        verbose=False
    )
    return X_train, y_train, sj_train

def load_numpy_datasets(DATASET_NAME="WISDM", dataset_folder="/home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/", USE_RECONSTRUCTED_DATA = False): # if use signal recontruction data or not (original data)
    load_path = f"{dataset_folder}numpies/train/"
    if USE_RECONSTRUCTED_DATA:
        load_path += "times-series-reconstruction/"
        X_train = np.load(f"{load_path}X_reconstructed.npy")
        y_train = np.load(f"{load_path}y_reconstructed.npy")
        sj_train = np.load(f"{load_path}sj_reconstructed.npy")
    else:
        load_path += "windowed/"
        X_train = np.load(f"{load_path}X_train.npy")
        y_train = np.load(f"{load_path}y_train.npy")
        sj_train = np.load(f"{load_path}sj_train.npy")

    y_train = to_categorical(y_train, len(np.unique(y_train)))
    print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
    return X_train, y_train, sj_train