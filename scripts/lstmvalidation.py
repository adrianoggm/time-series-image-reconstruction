#!C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/venv/bin/python3
from math import sqrt
import numpy as np
import src.ts_plots as plt
import src.activity_data as act
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(0, len(y)):
        for j in range(0, 5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y

def create_lstm_model(input_shape, num_classes=5):
    model = Sequential()
    # Primera capa LSTM
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    # Segunda capa LSTM
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    # Capa densa final
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    data_name = "WISDM"
    data_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
    
    # Cargamos los datos de entrenamiento
    X_train, y_train, sj_train = act.load_numpy_datasets(data_name, data_folder, USE_RECONSTRUCTED_DATA=False)
    print("X_train", X_train.shape, "y_train", y_train.shape, "sj_train", sj_train.shape)
    
    created = False  # Si True, se carga el modelo ya entrenado; si False, se entrena uno nuevo.
    stype = "GAF"   # Puede ser "RP", "GAF" o "MTF". Toma en cuenta las dimensiones.
    
    num_samples = X_train.shape[0]
    
    # Transformamos X_train para crear la entrada de 3 canales en formato (time_steps, features)
    if stype == "RP":
        # Para recurrence plot, cada serie tiene dimensión (3,128) -> se transponen a (128, 3)
        X_N = np.zeros((num_samples, 3, 128))
        for i in range(num_samples):
            X_N[i, 0, :] = X_train[i, :-1, 0]
            X_N[i, 1, :] = X_train[i, :-1, 1]
            X_N[i, 2, :] = X_train[i, :-1, 2]
        X_N = np.transpose(X_N, (0, 2, 1))  # (num_samples, 128, 3)
        input_shape = (128, 3)
    else:
        # Para GAF o MTF, cada serie tiene dimensión (3,129) -> se transponen a (129, 3)
        X_N = np.zeros((num_samples, 3, 129))
        for i in range(num_samples):
            X_N[i, 0, :] = X_train[i, :, 0]
            X_N[i, 1, :] = X_train[i, :, 1]
            X_N[i, 2, :] = X_train[i, :, 2]
        X_N = np.transpose(X_N, (0, 2, 1))  # (num_samples, 129, 3)
        input_shape = (129, 3)
    
    # Convertimos las etiquetas
    y = transformary(y_train)
    
    # Ruta para guardar/cargar el modelo
    model_path = "modelo_lstmRP.h5" if stype == "RP" else "modelo_lstmX.h5"
    
    if not created:
        # Creamos y entrenamos el modelo LSTM
        model = create_lstm_model(input_shape=input_shape, num_classes=5)
        print(model.summary())
        
        # Entrenamos el modelo
        model.fit(X_N, y, epochs=20, batch_size=64, validation_split=0.2)
        model.save(model_path)
    else:
        # Cargamos el modelo ya entrenado
        model = load_model(model_path)
    
    # Cargamos los datos de test
    data_folder_rp = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
    data_folder_gaf = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train"
    data_folder_mtf = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train"
    
    indices_T = np.load(f"{data_folder_rp}/y_training_data.npy")
    X_all_gaf = np.load(f"{data_folder_gaf}/X_all_gaf.npy")
    X_all_mtf = np.load(f"{data_folder_mtf}/X_all_mtf.npy")
    X_all_rec = np.load(f"{data_folder_rp}/X_all_rec.npy")
    
    # Seleccionamos y transformamos los datos de test según el stype
    if stype == "RP":
        # Suponemos que X_all_rec tiene forma (N, 3,128)
        X_test = np.transpose(X_all_rec, (0, 2, 1))  # (N,128,3)
    elif stype == "GAF":
        # Suponemos que X_all_gaf tiene forma (N, 3,129)
        X_test = np.transpose(X_all_gaf, (0, 2, 1))  # (N,129,3)
    elif stype == "MTF":
        # Suponemos que X_all_mtf tiene forma (N, 3,129)
        X_test = np.transpose(X_all_mtf, (0, 2, 1))  # (N,129,3)
    else:
        raise ValueError("stype desconocido")
    
    # Transformamos las etiquetas de test
    indices_T = transformary(indices_T)
    
    # Predecimos las etiquetas
    pred_probs = model.predict(X_test)
    prediccion = pred_probs.argmax(axis=-1)
    
    # Mostramos la matriz de confusión
    plt.plot_cm(indices_T, prediccion)
    
    # Imprimimos el reporte de clasificación
    print(classification_report(indices_T, prediccion))

if __name__ == '__main__':
    main()
