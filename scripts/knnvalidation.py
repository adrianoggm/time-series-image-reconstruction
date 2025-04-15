#!C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/venv/bin/python3
import os
import numpy as np
import src.ts_plots as plt
import src.activity_data as act
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import joblib

def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y

def flatten_series(X, stype):
    """
    Transforma las series 3D a una representación 2D (n_samples x (time_steps*canales)).
    Para 'RP', se descarta el último time step (de 129 se toman 128).
    Para 'GAF' o 'MTF', se asume la forma (129, 3).
    """
    if stype == "RP":
        X_proc = X[:, :-1, :]
    else:
        X_proc = X
    num_samples = X_proc.shape[0]
    return X_proc.reshape(num_samples, -1)

def main():
    # Flag: si True, se carga el modelo ya entrenado; si False, se entrena uno nuevo.
    created = True  
    # Selección de representación: "RP", "GAF" o "MTF"
    stype = "RP"  # Modifica según la técnica que quieras evaluar

    # Directorio base de los datos
    dataset_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"

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
    
    # Aplanamos las series según la técnica seleccionada
    X_train_flat = flatten_series(X_train, stype)
    X_val_flat   = flatten_series(X_val, stype)
    
    # Convertimos las etiquetas
    y_train_trans = transformary(y_train)
    y_val_trans   = transformary(y_val)
    
    # Definición de la ruta para guardar o cargar el modelo KNN
    
    model_path = os.path.join(dataset_folder, "modelo_knnX.joblib")
    
    # Entrenar o cargar el modelo KNN
    if not created:
        model = KNeighborsClassifier(n_neighbors=5)  # Puedes ajustar n_neighbors
        model.fit(X_train_flat, y_train_trans)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)
    
    # ---------------------------
    # 1. Predicción sobre validation_data.npy
    pred_val = model.predict(X_val_flat)
    print("Evaluación sobre validation_data.npy:")
    plt.plot_cm(y_val_trans, pred_val)
    print(classification_report(y_val_trans, pred_val))
    
    # ---------------------------
    # 2. Predicción sobre los datos reconstruidos
    # Definimos las rutas para los datos reconstruidos según la técnica
    data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")
    data_folder_gaf = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0")
    data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    
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
    else:
        raise ValueError("stype desconocido")
    
    # Aplanamos los datos reconstruidos
    num_samples = X_recon.shape[0]
    X_recon_flat = X_recon.reshape(num_samples, -1)
    
    pred_recon = model.predict(X_recon_flat)
    print("Evaluación sobre datos reconstruidos:")
    plt.plot_cm(y_val_trans, pred_recon)
    print(classification_report(y_val_trans, pred_recon))
    
if __name__ == '__main__':
    main()
