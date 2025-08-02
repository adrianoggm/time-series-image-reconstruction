import numpy as np
import src.calculate_ts_errors as cerr

# Cargar datos
data_folder="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
data_folder1="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train"
data_folder2="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train"
data_folder3="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GADF/sampling_loto/3-fold/fold-0/train"

indices_T=np.load(f"{data_folder}/training_data.npy")
X_all_gaf=np.load(f"{data_folder1}/X_all_gaf.npy")
X_all_mtf=np.load(f"{data_folder2}/X_all_mtf.npy")
X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
X_all_gadf=np.load(f"{data_folder3}/X_all_gadf.npy")
X_all_rec=np.transpose(X_all_rec, (0, 2, 1))

print("Testing with first sample...")
i = 0
w = indices_T[i]
print(f"Original data shape: {w.shape}")

# Test MTF
print("\nTesting MTF...")
rp = X_all_mtf[i]
print(f"MTF data shape: {rp.shape}")
try:
    error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp,flag=True)
    print("MTF: SUCCESS")
except Exception as e:
    print(f"MTF: ERROR - {e}")

# Test GAF
print("\nTesting GAF...")
rp = X_all_gaf[i]
print(f"GAF data shape: {rp.shape}")
try:
    error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp)
    print("GAF: SUCCESS")
except Exception as e:
    print(f"GAF: ERROR - {e}")

# Test REC
print("\nTesting REC...")
rp = X_all_rec[i]
print(f"REC data shape: {rp.shape}")
try:
    error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp)
    print("REC: SUCCESS")
except Exception as e:
    print(f"REC: ERROR - {e}")

# Test GADF
print("\nTesting GADF...")
rp = X_all_gadf[i]
print(f"GADF data shape: {rp.shape}")
try:
    error_abs,error_r,error_q,error_std,error_p,te=cerr.ts_error(w,rp)
    print("GADF: SUCCESS")
except Exception as e:
    print(f"GADF: ERROR - {e}")
