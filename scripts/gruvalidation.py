#!/usr/bin/env python3
import os
import numpy as np
import src.ts_plots as plt
import src.activity_data as act
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.config import DEFAULT_DATA_DIR

def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y

def create_gru_model(input_shape, num_classes=5):
    model = Sequential()
    # Primera capa GRU con salida secuencial para permitir apilar más capas
    model.add(GRU(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    # Segunda capa GRU
    model.add(GRU(32))
    model.add(Dropout(0.2))
    # Capa densa final para clasificación
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Flag: si True, se carga el modelo ya entrenado; si False, se entrena uno nuevo.
    created = True  
    # Selección de representación: "RP", "GAF" o "MTF"
    stype = "RP"  # Modifica según la técnica que quieras evaluar

    # Directorio base de los datos
    dataset_folder = DEFAULT_DATA_DIR

    # Rutas de los splits de entrenamiento y validación guardados previamente
    train_data_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/training_data.npy")
    train_y_path    = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/y_training_data.npy")
    val_data_path   = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/validation_data.npy")
    val_y_path      = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/y_validation_data.npy")
    
    # Cargamos los splits
    X_train = np.load(train_data_path)
    y_train = np.load(train_y_path)
    X_val   = np.load(val_data_path)
    y_val   = np.load(val_y_path)
    
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
    
    # Según el tipo de representación, transformamos los datos.
    # Para "GAF" o "MTF" se asume que las series ya tienen forma (129, 3)
    X_train_processed = X_train
    X_val_processed   = X_val
    input_shape = (129, 3)
    
    # Convertimos las etiquetas
    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)
    
    
    model_path = "modelo_gruX.h5"
    
    # Entrenar o cargar el modelo GRU
    if not created:
        model = create_gru_model(input_shape=input_shape, num_classes=5)
        print(model.summary())
        model.fit(X_train_processed, y_train_trans,
                  epochs=20,
                  batch_size=64,
                  validation_data=(X_val_processed, y_val_trans))
        model.save(model_path)
    else:
        model = load_model(model_path)
    
    # ---------------------------
    # 1. Predicción sobre validation_data.npy
    pred_probs_val = model.predict(X_val_processed)
    pred_val = pred_probs_val.argmax(axis=-1)
    print("Evaluación sobre validation_data.npy:")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val))
    
    # ---------------------------
    # 2. Predicción sobre los datos reconstruidos
    # Definimos las rutas para cada técnica; se asume que los archivos reconstruidos están en la carpeta "test"
    data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")
    data_folder_gaf = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0")
    data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    data_folder_gadf = os.path.join(dataset_folder, "tseries/GADF/sampling_loto/3-fold/fold-0")
    
    if stype == "RP":
        rec_path = os.path.join(data_folder_rp, "test", "X_all_rec.npy")
        X_all_rec = np.load(rec_path)
        # Se asume que X_all_rec tiene forma (N, 3, 128); se transpone a (N, 128, 3)
        X_recon = X_all_rec#np.transpose(X_all_rec, (0, 2, 1))
    elif stype == "GAF":
        gaf_path = os.path.join(data_folder_gaf, "test", "X_all_gaf.npy")
        X_all_gaf = np.load(gaf_path)
        # Se asume que X_all_gaf tiene forma (N, 3, 129)
        X_recon = np.transpose(X_all_gaf, (0, 2, 1))
    elif stype == "MTF":
        mtf_path = os.path.join(data_folder_mtf, "test", "X_all_mtf.npy")
        X_all_mtf = np.load(mtf_path)
        # Se asume que X_all_mtf tiene forma (N, 3, 129)
        X_recon = np.transpose(X_all_mtf, (0, 2, 1))
    elif stype == "GADF":
        gadf_path = os.path.join(data_folder_gadf, "test", "X_all_gadf.npy")
        X_all_gadf = np.load(gadf_path)
        # Forma esperada: (N, 3, 129)
        X_recon = np.transpose(X_all_gadf, (0, 2, 1))
    else:
        raise ValueError("stype desconocido")
    
    # Realizamos la predicción sobre las reconstrucciones
    pred_probs_recon = model.predict(X_recon)
    pred_recon = pred_probs_recon.argmax(axis=-1)
    print("Evaluación sobre datos reconstruidos:")
    plt.plot_cm(y_val_trans, pred_recon)
    print(classification_report(y_val_trans, pred_recon))
    
if __name__ == '__main__':
    main()
