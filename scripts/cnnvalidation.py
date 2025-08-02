#!C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/venv/bin/python3
import os
import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Dense, Dropout, Concatenate, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
import src.ts_plots as plt

def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        # Itera según la cantidad de columnas disponibles
        for j in range(y.shape[1]):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y

def compute_additional_features(x, bins=10):
    """
    Calcula, para cada muestra, las siguientes características por canal:
      - media
      - varianza
      - suma de valores absolutos
      - histograma (con 'bins' valores)
    Devuelve un vector de tamaño: channels * (3 + bins)
    x tiene forma [time_steps, channels].
    """
    feats = []
    for ch in range(x.shape[1]):
        channel_data = x[:, ch]
        m = np.mean(channel_data)
        var = np.var(channel_data)
        s = np.sum(np.abs(channel_data))
        min_val, max_val = np.min(channel_data), np.max(channel_data)
        if min_val == max_val:
            hist = np.zeros(bins, dtype=np.float32)
        else:
            hist, _ = np.histogram(channel_data, bins=bins, range=(min_val, max_val))
            hist = hist.astype(np.float32)
        feats.extend([m, var, s])
        feats.extend(hist.tolist())
    return np.array(feats, dtype=np.float32)

def additional_features_layer(x, bins=10):
    """
    Aplica la función de características adicionales a cada muestra.
    x tiene forma (batch, time_steps, channels) y se devuelve (batch, channels*(3+bins)).
    """
    return tf.py_function(
        func=lambda batch: tf.stack([
            tf.py_function(compute_additional_features, [sample, bins], tf.float32)
            for sample in tf.unstack(batch)
        ]),
        inp=[x, bins],
        Tout=tf.float32
    )

def create_cnn_model(input_shape, num_classes=5, l2_reg=0.001):
    """
    Crea un modelo CNN simplificado con Conv1D.
    """
    inp = Input(shape=input_shape)  # Ejemplo: (128, 3)
    
    # Primera capa convolucional
    conv1 = Conv1D(filters=64, kernel_size=8, strides=1, activation='relu',
                   kernel_regularizer=l2(l2_reg))(inp)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    
    # Segunda capa convolucional
    conv2 = Conv1D(filters=128, kernel_size=4, strides=1, activation='relu',
                   kernel_regularizer=l2(l2_reg))(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)
    
    # Tercera capa convolucional
    conv3 = Conv1D(filters=196, kernel_size=4, strides=1, activation='relu',
                   kernel_regularizer=l2(l2_reg))(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)
    
    # Aplanar y capas densas
    flat = Flatten()(pool3)
    fc1 = Dense(512, activation='relu', kernel_regularizer=l2(l2_reg))(flat)
    fc1 = Dropout(0.3)(fc1)
    fc2 = Dense(256, activation='relu', kernel_regularizer=l2(l2_reg))(fc1)
    fc2 = Dropout(0.3)(fc2)
    out = Dense(num_classes, activation='softmax')(fc2)
    
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run classifier validation')
    parser.add_argument('--technique', type=str, default='GAF',
                       choices=['GAF', 'GADF', 'RP', 'MTF'],
                       help='Reconstruction technique to use')
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting CNN validation script...")
    args = parse_arguments()
    stype = args.technique
    print(f"Technique selected: {stype}")
    
    dataset_folder = "data/WISDM"
    data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")
    data_folder_gaf = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0")
    data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    data_folder_gadf = os.path.join(dataset_folder, "tseries/GADF/sampling_loto/3-fold/fold-0")
    
    # Load original training and validation data
    train_x_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train", "training_data.npy")
    train_y_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train", "y_training_data.npy")
    val_x_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test", "validation_data.npy")
    val_y_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test", "y_validation_data.npy")

    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    X_val = np.load(val_x_path)
    y_val = np.load(val_y_path)

    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)

    # Convert labels to class indices
    y_train_idx = transformary(y_train)
    y_val_idx = transformary(y_val)

    # Evaluate on original data
    print("Evaluación sobre validation_data.npy: ORIGINALES")

    model_filename = "modelo_cnn_original.h5"
    if os.path.exists(model_filename):
        print(f"Loading existing model: {model_filename}")
        model = load_model(model_filename)
    else:
        print(f"Model {model_filename} not found. Training new CNN model with original data")
        model = create_cnn_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train_idx, 
                  validation_split=0.2, 
                  epochs=50, 
                  batch_size=32, 
                  verbose=1)
        model.save(model_filename)
        print(f"Model saved as: {model_filename}")

    # Predict on original validation data
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_val_idx, y_pred_classes))

    # Load and evaluate reconstructed data
    if stype == "RP":
        rec_path = os.path.join(data_folder_rp, "test", "X_all_rec.npy")
        X_all_rec = np.load(rec_path)
        X_recon = np.transpose(X_all_rec, (0, 2, 1))
        print("X_recon RP shape:", X_all_rec.shape, "-> transposed:", X_recon.shape)
    elif stype == "GAF":
        gaf_path = os.path.join(data_folder_gaf, "test", "X_all_gaf.npy")
        X_all_gaf = np.load(gaf_path)
        X_recon = np.transpose(X_all_gaf, (0, 2, 1))
    elif stype == "GADF":
        gadf_path = os.path.join(data_folder_gadf, "test", "X_all_gadf.npy")
        X_all_gadf = np.load(gadf_path)
        X_recon = np.transpose(X_all_gadf, (0, 2, 1))
        print("X_recon GADF shape:", X_all_gadf.shape, "-> transposed:", X_recon.shape)
    elif stype == "MTF":
        mtf_path = os.path.join(data_folder_mtf, "test", "X_all_mtf.npy")
        X_all_mtf = np.load(mtf_path)
        X_recon = np.transpose(X_all_mtf, (0, 2, 1))
    else:
        raise ValueError(f"stype desconocido: {stype}")

    print(f"Evaluación sobre datos reconstruidos: {stype}")
    y_pred_recon = model.predict(X_recon)
    y_pred_recon_classes = np.argmax(y_pred_recon, axis=1)
    print(classification_report(y_val_idx, y_pred_recon_classes))
