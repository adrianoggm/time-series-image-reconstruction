#!C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/venv/bin/python3
import os
import numpy as np
import argparse
import src.ts_plots as plt
import src.activity_data as act
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def transformary(y):
    _y = np.ones(len(y)) * -1
    for i in range(len(y)):
        for j in range(5):
            if y[i, j] <= 1.1 and y[i, j] >= 0.9:
                _y[i] = j
    return _y

def create_lstm_model(input_shape, num_classes=5):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run LSTM classifier validation')
    parser.add_argument('--technique', type=str, default='GAF',
                       choices=['GAF', 'GADF', 'RP', 'MTF'],
                       help='Reconstruction technique to use')
    return parser.parse_args()

# Main execution
args = parse_arguments()
stype = args.technique

# Dataset configuration
dataset_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")
data_folder_gaf = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0")
data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
data_folder_gadf = os.path.join(dataset_folder, "tseries/GADF/sampling_loto/3-fold/fold-0")

# Load training and validation data
train_data_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/training_data.npy")
train_y_path    = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/y_training_data.npy")
val_data_path   = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/validation_data.npy")
val_y_path      = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/y_validation_data.npy")

X_train = np.load(train_data_path)
y_train = np.load(train_y_path)
X_val = np.load(val_data_path)
y_val = np.load(val_y_path)

print("X_train shape:", X_train.shape, "y_train shape:", y_train.shape)
print("X_val shape:", X_val.shape, "y_val shape:", y_val.shape)

# Convert labels to class indices
y_train_idx = transformary(y_train)
y_val_idx = transformary(y_val)

# Evaluate on original data
print("Evaluación sobre validation_data.npy: ORIGINALES")

model_filename = "modelo_lstm_original.h5"
if os.path.exists(model_filename):
    print(f"Loading existing model: {model_filename}")
    model = load_model(model_filename)
else:
    print(f"Model {model_filename} not found. Training new LSTM model with original data")
    model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
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