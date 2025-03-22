#!C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/venv/bin/python3
import os
import numpy as np
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
    def _compute_features(sample):
        return compute_additional_features(sample, bins=bins)
    features = tf.map_fn(lambda samp: tf.py_function(_compute_features, [samp], tf.float32),
                           x, fn_output_signature=tf.float32)
    channels = tf.shape(x)[-1]
    features.set_shape([None, 3*(3+bins)])
    return features

def create_cnn_model_paper(input_shape, num_classes=6, l2_reg=0.001, bins=10):
    """
    Crea la CNN inspirada en el paper:
      - Conv1D con 196 filtros de tamaño 16, stride 1, activación ReLU y regularización L2.
      - MaxPooling1D (pool_size=4).
      - Se aplana la salida del pooling.
      - Se calculan características estadísticas adicionales (media, varianza, suma y histograma por canal) a partir de la entrada original.
      - Se concatenan ambos vectores y se pasan a una capa FC de 1024 neuronas (con dropout 0.05).
      - Capa softmax para clasificación.
    """
    inp = Input(shape=input_shape)  # Ejemplo: (128, 3) o (129, 3)
    conv = Conv1D(filters=196, kernel_size=16, strides=1, activation='relu',
                  kernel_regularizer=l2(l2_reg))(inp)
    pool = MaxPooling1D(pool_size=4)(conv)
    flat = Flatten()(pool)
    stats = Lambda(lambda x: additional_features_layer(x, bins=bins))(inp)
    concat = Concatenate()([flat, stats])
    fc = Dense(1024, activation='relu', kernel_regularizer=l2(l2_reg))(concat)
    fc = Dropout(0.05)(fc)
    out = Dense(num_classes, activation='softmax')(fc)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Flag para decidir si se entrena un modelo nuevo o se carga uno existente
    created = False
    # Selección de representación: "RP", "GAF" o "MTF"
    stype = "GAF"  # Si usas GAF o MTF se asume que la forma es (129, 3)
    
    # Directorio base de los datos
    dataset_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
    
    # Rutas de splits de entrenamiento y validación
    train_data_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/training_data.npy")
    train_y_path    = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/y_training_data.npy")
    val_data_path   = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/validation_data.npy")
    val_y_path      = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/y_validation_data.npy")
    
    # Cargamos los datos
    X_train = np.load(train_data_path)
    y_train = np.load(train_y_path)
    X_val   = np.load(val_data_path)
    y_val   = np.load(val_y_path)
    
    print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
    print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)
    
    # Transformación de datos según la técnica.
    # Para "RP": se descarta el último time step (de 129 se toman 128)
    if stype == "RP":
        num_train = X_train.shape[0]
        X_train_processed = np.zeros((num_train, 128, 3))
        for i in range(num_train):
            X_train_processed[i, :, 0] = X_train[i, :-1, 0]
            X_train_processed[i, :, 1] = X_train[i, :-1, 1]
            X_train_processed[i, :, 2] = X_train[i, :-1, 2]
        num_val = X_val.shape[0]
        X_val_processed = np.zeros((num_val, 128, 3))
        for i in range(num_val):
            X_val_processed[i, :, 0] = X_val[i, :-1, 0]
            X_val_processed[i, :, 1] = X_val[i, :-1, 1]
            X_val_processed[i, :, 2] = X_val[i, :-1, 2]
        input_shape = (128, 3)
    else:
        X_train_processed = X_train
        X_val_processed = X_val
        input_shape = (129, 3)
    
    # Convertimos las etiquetas (se esperan 6 clases)
    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)
    
    # Ruta para guardar o cargar el modelo
    if stype == "RP":
        model_path = os.path.join(dataset_folder, "modelo_cnnPaperRP.h5")
    else:
        model_path = os.path.join(dataset_folder, "modelo_cnnPaperX.h5")
    
    if not created:
        model = create_cnn_model_paper(input_shape=input_shape, num_classes=6, l2_reg=0.0001, bins=10)
        print(model.summary())
        model.fit(X_train_processed, y_train_trans,
                  epochs=20,
                  batch_size=64,
                  validation_data=(X_val_processed, y_val_trans))
        model.save(model_path)
    else:
        model = load_model(model_path, custom_objects={'compute_additional_features': compute_additional_features})
    
    # ---------------------------
    # 1. Predicción sobre validation_data.npy
    pred_probs_val = model.predict(X_val_processed)
    pred_val = pred_probs_val.argmax(axis=-1)
    print("Evaluación sobre validation_data.npy:")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val))
    
    # ---------------------------
    # 2. Predicción sobre los datos reconstruidos
    data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")
    data_folder_gaf = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0")
    data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    
    if stype == "RP":
        rec_path = os.path.join(data_folder_rp, "test", "X_all_rec.npy")
        X_all_rec = np.load(rec_path)
        X_recon = np.transpose(X_all_rec, (0, 2, 1))
    elif stype == "GAF":
        gaf_path = os.path.join(data_folder_gaf, "test", "X_all_gaf.npy")
        X_all_gaf = np.load(gaf_path)
        X_recon = np.transpose(X_all_gaf, (0, 2, 1))
    elif stype == "MTF":
        mtf_path = os.path.join(data_folder_mtf, "test", "X_all_mtf.npy")
        X_all_mtf = np.load(mtf_path)
        X_recon = np.transpose(X_all_mtf, (0, 2, 1))
    else:
        raise ValueError("stype desconocido")
    
    pred_probs_recon = model.predict(X_recon)
    pred_recon = pred_probs_recon.argmax(axis=-1)
    print("Evaluación sobre datos reconstruidos:")
    plt.plot_cm(y_val_trans, pred_recon)
    print(classification_report(y_val_trans, pred_recon))
    
if __name__ == '__main__':
    main()
