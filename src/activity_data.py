import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical

from src.config import DEFAULT_DATA_DIR


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

def get_time_setup(DATASET_NAME="WISDM"):
    # In WISDM, since sampling rate is 20Hz: 20Hz * seconds of duration
    if DATASET_NAME == "WISDM" or DATASET_NAME == "ORIGINAL_WISDM":
        DATASET_HZ = 20  # WISDM 20Hz; ADL_Dataset 32Hz
        TIME_SECS = 6.4  # in secs
    elif DATASET_NAME == "ADL_Dataset": 
        DATASET_HZ = 32
        TIME_SECS = 10  # in secs
    elif "MINDER":
        DATASET_HZ = 4  # 4 hours
        TIME_SECS = 60  # in secs
    else:
        DATASET_HZ = 32

    # Window length in samples. We use a power-of-two (128 for WISDM @ 20 Hz)
    # because most CNN/RNN tooling assumes that, and because it keeps the
    # final RP image at (N-1) x (N-1) = 127 x 127 while GAF/GADF/MTF stay at
    # N x N = 128 x 128. The old "+1 to make RPs 128x128" workaround is gone
    # — see `src.recurrence_plots.fix_rotationscale`, which now augments the
    # MDS-recovered series (N-1 samples) back to N by appending the mean.
    TIME_STEPS = int(DATASET_HZ * TIME_SECS)
    STEP = TIME_STEPS  # without overlapping; 50% overlaps == TIME_STEPS//2
    print("dataset:", DATASET_NAME, "Sampling rate:", DATASET_HZ, "samples per window:", TIME_STEPS)
    return TIME_STEPS, STEP


def create_wisdm_dataset(path):
    activities = {
        "A": "Walking", "B": "Jogging", "C": "Stairs", "D": "Sitting",
        "E": "Standing", "F": "Typing", "G": "Brushing Teeth", "H": "Eating Soup",
        "I": "Eating Chips", "J": "Eating Pasta", "K": "Drinking from Cup",
        "L": "Eating Sandwich", "M": "Kicking (Soccer Ball)", "O": "Playing Catch w/Tennis Ball",
        "P": "Dribbling (Basketball)", "Q": "Writting", "R": "Clapping", "S": "Folding Clothes"
    }
    df = pd.DataFrame()
    for f in sorted(os.listdir(path)):
        daux = pd.read_csv(os.path.join(path, f), header=None, 
                           names=['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis'])
        daux.activity = daux.activity.apply(lambda x: activities[x])
        # Evitamos el chained assignment asignando el resultado de replace a la columna 'z_axis'
        daux["z_axis"] = daux["z_axis"].replace(regex=True, to_replace=r';', value=r'')
        daux['z_axis'] = daux['z_axis'].astype(np.float64)
        daux.dropna(axis=0, how='any', inplace=True)
        df = pd.concat([df, daux])
    return df

def preprocess_activity_data_sets(DATASET_NAME="WISDM", dataset_folder=DEFAULT_DATA_DIR,
                                  column_names=['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']):

    if DATASET_NAME == "MINDER":
        df = pd.read_csv(f"{dataset_folder}minder-preprocessed.csv")
        df = df.rename(columns={"uti_label": "activity", "patient_id": "user_id"})
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
        df = pd.read_csv(f'{dataset_folder}WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt', header=None, names=column_names)
        # Se evita chained assignment
        df["z_axis"] = df["z_axis"].replace(regex=True, to_replace=r';', value=r'')
        df['z_axis'] = df['z_axis'].astype(np.float64)
        df.dropna(axis=0, how='any', inplace=True)
    elif DATASET_NAME == "ADL_Dataset":
        adls = ["Climb_stairs", "Drink_glass", "Getup_bed", "Pour_water", "Sitdown_chair", "Standup_chair", "Walk"]
        dfs = []
        activity_labels = []
        final_ids = []
        user_ids = []
        timestamps = []
        for adl in adls:
            path = f'ADL_Dataset/HMP_Dataset/{adl}/'
            print(f"Processing {adl}")
            files = os.listdir(path)
            for f in files:
                id = f.split("-")[8].split(".")[0]
                if id not in final_ids:
                    final_ids.append(id)
                df_temp = pd.read_csv(os.path.join(path, f), sep=" ", header=None)
                dfs.append(df_temp)
                timestamps.extend(list(range(len(df_temp))))
                user_ids.extend(np.repeat(final_ids.index(id), len(df_temp)))
                activity_labels.extend(np.repeat(adl, len(df_temp)))
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


import shutil

def create_dataset(savepath, data_type, X, y, sj_id, time_steps=1, step=1, verbose=False):
    """Segment-aware sliding-window segmentation.

    Windows never cross a (subject, activity) boundary. The concatenated stream is
    split into maximal contiguous runs of constant (subject, activity), and
    non-overlapping windows of ``time_steps`` samples are taken *within* each run;
    the tail shorter than ``time_steps`` is dropped. This removes the boundary
    windows produced by the previous global slide, which mixed two labels/subjects
    per window and leaked across subjects in the subject-out (LOSO) protocol.
    """
    Xv = X.values
    yv = np.asarray(y.values).reshape(-1)
    sjv = np.asarray(sj_id.values).reshape(-1)
    n = len(Xv)

    Xs_all, ys_all, sjs_all = [], [], []
    per_group: dict = {}

    start = 0
    while start < n:
        end = start + 1
        while end < n and sjv[end] == sjv[start] and yv[end] == yv[start]:
            end += 1
        label = yv[start]
        subject = sjv[start]
        if isinstance(label, str):
            label = label.replace(" ", "-").replace("/", "-").replace("(", "").replace(")", "")
        # non-overlapping windows fully inside the run; tail < time_steps dropped
        for i in range(start, end - time_steps + 1, step):
            v = Xv[i:i + time_steps]
            if (v == 0).all():
                continue
            Xs_all.append(v)
            ys_all.append(label)
            sjs_all.append(subject)
            per_group.setdefault((label, subject), []).append(v)
        start = end

    base = f"{savepath}{data_type}/windowed/"
    shutil.rmtree(base, ignore_errors=True)
    os.makedirs(base, exist_ok=True)
    for (label, subject), wins in per_group.items():
        os.makedirs(f"{base}{label}/{subject}/", exist_ok=True)
        np.save(f"{base}{label}/{subject}/x.npy", np.array(wins))

    X_out = np.array(Xs_all)
    y_out = np.array(ys_all).reshape(-1, 1)
    sj_out = np.array(sjs_all).reshape(-1, 1)
    np.save(f"{base}X_{data_type}.npy", X_out)
    np.save(f"{base}y_{data_type}.npy", y_out)
    np.save(f"{base}sj_{data_type}.npy", sj_out)
    print(f"[windowing] {data_type}: {len(X_out)} windows over {len(per_group)} "
          f"(subject, activity) segments (segment-aware; no boundary crossing)")
    return X_out, y_out, sj_out


def create_all_numpy_datasets(DATASET_NAME="WISDM", dataset_folder=DEFAULT_DATA_DIR,
                              COL_SELECTED_IDXS=list(range(3, 3+3))):
    TIME_STEPS, STEP = get_time_setup(DATASET_NAME)
    df = preprocess_activity_data_sets(DATASET_NAME, dataset_folder, 
                                       column_names=['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis'])
    print("Classes preprocessed:", np.unique(df.activity))
    X_train, y_train, sj_train = create_dataset(
        dataset_folder + "numpies/",
        "train",
        df.iloc[:, COL_SELECTED_IDXS], 
        df.activity,
        df.user_id,
        TIME_STEPS, 
        STEP,
        verbose=False
    )
    return X_train, y_train, sj_train

def load_numpy_datasets(DATASET_NAME="WISDM", dataset_folder=DEFAULT_DATA_DIR,
                        USE_RECONSTRUCTED_DATA=False):
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
