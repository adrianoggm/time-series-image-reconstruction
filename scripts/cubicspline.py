import os
import numpy as np
from scipy.interpolate import CubicSpline
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Global style matching comparative plots
sns.set(style="whitegrid", context="paper", font="serif", rc={
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "lines.linewidth": 1.5,
    "grid.color": "0.8",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
})

def plot_rp_sample(original, upsampled, sample_index=0, channel=0, save_folder=None):
    """
    Plot a sample from the RP series, showing original and upsampled versions
    using the same style as the comparative plots.
    """
    t_orig = np.linspace(0, 1, original.shape[1])
    t_up = np.linspace(0, 1, upsampled.shape[1])
    
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(t_orig, original[sample_index, :, channel],
            'o', label="Original RP (128 pts)", markersize=4, color="C0")
    ax.plot(t_up, upsampled[sample_index, :, channel],
            '-', label="Upsampled RP (129 pts)", color="C1")
    
    ax.set_title(f"Recurrence Plot Sample — Sample {sample_index}, Channel {channel}")
    ax.set_xlabel("Normalized Time")
    ax.set_ylabel("Recurrence Value")
    # Place legend outside
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    ax.grid(True)
    fig.tight_layout(rect=[0, 0, 0.78, 1])  # adjust right margin
    
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder,
                                 f"rp_sample_{sample_index}_ch{channel}.png")
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Image saved to: {save_path}")
    plt.close(fig)


def upsample_series(data, new_n_points):
    """
    Remuestrea la serie a un nuevo número de puntos usando interpolación spline.
    
    Parámetros:
      data: np.ndarray de forma (n_samples, n_points, n_channels)
      new_n_points: int, el nuevo número de puntos deseados.
      
    Retorna:
      data_upsampled: np.ndarray de forma (n_samples, new_n_points, n_channels)
    """
    n_samples, n_points, n_channels = data.shape
    t_original = np.linspace(0, 1, n_points)
    t_new = np.linspace(0, 1, new_n_points)
    
    data_upsampled = np.zeros((n_samples, new_n_points, n_channels))
    for i in range(n_samples):
        for ch in range(n_channels):
            cs = CubicSpline(t_original, data[i, :, ch], bc_type='natural')
            data_upsampled[i, :, ch] = cs(t_new)
    return data_upsampled

def compute_spline_coefficients(data, t):
    """
    Calcula los coeficientes del spline cúbico para cada muestra y canal.
    
    Parámetros:
      data: np.ndarray de forma (n_samples, n_points, n_channels)
      t: vector de tiempo de longitud n_points, asumiendo puntos igualmente espaciados.
      
    Retorna:
      coeffs_all: np.ndarray de forma (n_samples, n_channels, 4*(n_points-1))
                  con los coeficientes aplanados para cada spline.
    """
    n_samples, n_points, n_channels = data.shape
    coeffs_all = np.zeros((n_samples, n_channels, 4 * (n_points - 1)))
    for i in range(n_samples):
        for ch in range(n_channels):
            y = data[i, :, ch]
            cs = CubicSpline(t, y, bc_type='natural')
            coeffs_all[i, ch, :] = cs.c.flatten()
    return coeffs_all

def normalize_coeffs(coeffs):
    """
    Normaliza cada vector de coeficientes (por muestra y canal) usando la norma L2.
    
    Parámetros:
      coeffs: np.ndarray de forma (n_samples, n_channels, vector_size)
      
    Retorna:
      coeffs_norm: np.ndarray con la misma forma que coeffs, donde cada vector se ha normalizado.
    """
    coeffs_norm = np.copy(coeffs)
    n_samples, n_channels, _ = coeffs.shape
    for i in range(n_samples):
        for ch in range(n_channels):
            vec = coeffs[i, ch, :]
            norm = np.linalg.norm(vec)
            if norm > 0:
                coeffs_norm[i, ch, :] = vec / norm
    return coeffs_norm

def compute_metrics(coeffs_ref, coeffs_comp):
    """
    Calcula métricas de comparación entre dos conjuntos de coeficientes para cada muestra y canal:
      - Distancia Euclidiana.
      - Distancia Manhattan.
      - Similitud del coseno.
      - Distancia de Mahalanobis (usando la covarianza de las diferencias).
    
    Parámetros:
      coeffs_ref: np.ndarray de forma (n_samples, n_channels, vector_size)
      coeffs_comp: np.ndarray de forma (n_samples, n_channels, vector_size)
    
    Retorna:
      Una tupla con (promedio Euclidiana, promedio Manhattan, promedio Coseno, promedio Mahalanobis).
    """
    n_samples, n_channels, _ = coeffs_ref.shape
    euclidean_distances = []
    manhattan_distances = []
    cosine_similarities = []
    diff_vectors = []  # para Mahalanobis

    for i in range(n_samples):
        for ch in range(n_channels):
            v_ref = coeffs_ref[i, ch, :]
            v_comp = coeffs_comp[i, ch, :]
            diff = v_ref - v_comp
            diff_vectors.append(diff)
            euclidean_distances.append(np.linalg.norm(diff))
            manhattan_distances.append(np.sum(np.abs(diff)))
            cosine_sim = cosine_similarity(v_ref.reshape(1, -1), v_comp.reshape(1, -1))[0, 0]
            cosine_similarities.append(cosine_sim)

    diff_vectors = np.array(diff_vectors)  # (n_samples*n_channels, vector_size)
    cov = np.cov(diff_vectors, rowvar=False)
    inv_cov = np.linalg.pinv(cov)
    mahalanobis_distances = []
    for diff in diff_vectors:
        d = np.sqrt(diff.T.dot(inv_cov).dot(diff))
        mahalanobis_distances.append(d)

    avg_euclidean = np.mean(euclidean_distances)
    avg_manhattan = np.mean(manhattan_distances)
    avg_cosine = np.mean(cosine_similarities)
    avg_mahalanobis = np.mean(mahalanobis_distances)

    return avg_euclidean, avg_manhattan, avg_cosine, avg_mahalanobis

def compute_pearson_coeff(x, y):
    """
    Calcula el coeficiente de correlación de Pearson para cada muestra y canal entre
    dos series x e y (con forma: n_samples, n_points, n_channels). 
    
    Retorna:
      channel_mean: array de tamaño (n_channels,) con el promedio de Pearson por canal.
      overall_mean: promedio global (todos los canales y muestras).
    """
    n_samples, n_points, n_channels = x.shape
    corr_all = np.zeros((n_samples, n_channels))
    for i in range(n_samples):
        for ch in range(n_channels):
            r = np.corrcoef(x[i, :, ch], y[i, :, ch])[0, 1]
            corr_all[i, ch] = r
    channel_mean = np.mean(corr_all, axis=0)
    overall_mean = np.mean(corr_all)
    return channel_mean, overall_mean


def main():
    # Define las rutas a tus datos
    dataset_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
    images_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/images"

    train_data_path = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train/training_data.npy")
    series_rp_path   = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/X_all_rec.npy")
    series_rp_path2   = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0/test/X_all_rec2.npy")
    series_gaf_path  = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0/train/X_all_gaf.npy")
    series_mtf_path  = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0/train/X_all_mtf.npy")

    # Carga las series
    X_train = np.load(train_data_path)  # (n_samples, 129, 3)
    X_rp    = np.load(series_rp_path)    # (n_samples, 3, 128)
    X_gaf   = np.load(series_gaf_path)   # (n_samples, 3, 129)
    X_mtf   = np.load(series_mtf_path)   # (n_samples, 3, 129)

    print("Shape de X_train:", X_train.shape)
    

    # Reordena las reconstrucciones para que tengan la forma (n_samples, puntos, canales)
    X_rp = X_rp#np.transpose(X_rp, (0, 2, 1))   # (n_samples, 128, 3)
    X_gaf = np.transpose(X_gaf, (0, 2, 1)) # (n_samples, 129, 3)
    X_mtf = np.transpose(X_mtf, (0, 2, 1)) # (n_samples, 129, 3)
    print("Shape de X_rp:", X_rp.shape)
    print("Shape de X_gaf:", X_gaf.shape)
    print("Shape de X_mtf:", X_mtf.shape)
    # Parámetros para X_train, GAF y MTF (129 puntos)
    n_samples, n_points_train, n_channels = X_train.shape
    t_train = np.linspace(0, 1, n_points_train)
    """
        # Para RP: originalmente tiene 128 puntos, se remuestrea a 129 puntos
        X_rp_upsampled = upsample_series(X_rp, n_points_train)  # ahora (n_samples, 129, 3)
        
        # Guardamos el RP upsampleado en el mismo path que el original X_all_rec.npy
        np.save(series_rp_path2, X_rp_upsampled)
        print("Archivo X_rp_upsampled", X_rp_upsampled.shape, "guardado en:", series_rp_path2)

        t_rp_new = t_train  # coincide con 129 puntos
    """
    # Calcula los coeficientes de spline para cada conjunto
    coeffs_train = compute_spline_coefficients(X_train, t_train)
    coeffs_rp = compute_spline_coefficients(X_rp, t_train)
    coeffs_gaf = compute_spline_coefficients(X_gaf, t_train)
    coeffs_mtf = compute_spline_coefficients(X_mtf, t_train)

    # Normaliza los coeficientes
    coeffs_train_norm = normalize_coeffs(coeffs_train)
    coeffs_rp_norm = normalize_coeffs(coeffs_rp)
    coeffs_gaf_norm = normalize_coeffs(coeffs_gaf)
    coeffs_mtf_norm = normalize_coeffs(coeffs_mtf)

    # Calcula las métricas con los coeficientes originales
    metrics_rp = compute_metrics(coeffs_train, coeffs_rp)
    metrics_gaf = compute_metrics(coeffs_train, coeffs_gaf)
    metrics_mtf = compute_metrics(coeffs_train, coeffs_mtf)

    # Calcula las métricas con los coeficientes normalizados
    metrics_rp_norm = compute_metrics(coeffs_train_norm, coeffs_rp_norm)
    metrics_gaf_norm = compute_metrics(coeffs_train_norm, coeffs_gaf_norm)
    metrics_mtf_norm = compute_metrics(coeffs_train_norm, coeffs_mtf_norm)

    # Desempaqueta las métricas: (Euclidiana, Manhattan, Coseno, Mahalanobis)
    euclid_rp, manhat_rp, cosine_rp, maha_rp = metrics_rp
    euclid_gaf, manhat_gaf, cosine_gaf, maha_gaf = metrics_gaf
    euclid_mtf, manhat_mtf, cosine_mtf, maha_mtf = metrics_mtf

    euclid_rp_norm, manhat_rp_norm, cosine_rp_norm, maha_rp_norm = metrics_rp_norm
    euclid_gaf_norm, manhat_gaf_norm, cosine_gaf_norm, maha_gaf_norm = metrics_gaf_norm
    euclid_mtf_norm, manhat_mtf_norm, cosine_mtf_norm, maha_mtf_norm = metrics_mtf_norm

    print("Métricas CON COEFICIENTES ORIGINALES:")
    print("RP:")
    print("  Distancia Euclidiana promedio:", euclid_rp)
    print("  Distancia Manhattan promedio:", manhat_rp)
    print("  Similitud del coseno promedio:", cosine_rp)
    print("  Distancia de Mahalanobis promedio:", maha_rp)
    print("\nGAF:")
    print("  Distancia Euclidiana promedio:", euclid_gaf)
    print("  Distancia Manhattan promedio:", manhat_gaf)
    print("  Similitud del coseno promedio:", cosine_gaf)
    print("  Distancia de Mahalanobis promedio:", maha_gaf)
    print("\nMTF:")
    print("  Distancia Euclidiana promedio:", euclid_mtf)
    print("  Distancia Manhattan promedio:", manhat_mtf)
    print("  Similitud del coseno promedio:", cosine_mtf)
    print("  Distancia de Mahalanobis promedio:", maha_mtf)

    print("\nMétricas CON COEFICIENTES NORMALIZADOS:")
    print("RP:")
    print("  Distancia Euclidiana promedio:", euclid_rp_norm)
    print("  Distancia Manhattan promedio:", manhat_rp_norm)
    print("  Similitud del coseno promedio:", cosine_rp_norm)
    print("  Distancia de Mahalanobis promedio:", maha_rp_norm)
    print("\nGAF:")
    print("  Distancia Euclidiana promedio:", euclid_gaf_norm)
    print("  Distancia Manhattan promedio:", manhat_gaf_norm)
    print("  Similitud del coseno promedio:", cosine_gaf_norm)
    print("  Distancia de Mahalanobis promedio:", maha_gaf_norm)
    print("\nMTF:")
    print("  Distancia Euclidiana promedio:", euclid_mtf_norm)
    print("  Distancia Manhattan promedio:", manhat_mtf_norm)
    print("  Similitud del coseno promedio:", cosine_mtf_norm)
    print("  Distancia de Mahalanobis promedio:", maha_mtf_norm)
    
    # Calcula el coeficiente de correlación de Pearson para cada canal y globalmente
    pearson_rp_channel, pearson_rp_overall = compute_pearson_coeff(X_train, X_rp)
    pearson_gaf_channel, pearson_gaf_overall = compute_pearson_coeff(X_train, X_gaf)
    pearson_mtf_channel, pearson_mtf_overall = compute_pearson_coeff(X_train, X_mtf)
    
    print("\nCoeficiente de correlación de Pearson:")
    print("RP por canal:", pearson_rp_channel)
    print("RP promedio global:", pearson_rp_overall)
    print("GAF por canal:", pearson_gaf_channel)
    print("GAF promedio global:", pearson_gaf_overall)
    print("MTF por canal:", pearson_mtf_channel)
    print("MTF promedio global:", pearson_mtf_overall)
    
    # Grafica y guarda una muestra de RP: original vs. upsampleada.
    # Se toma la muestra 0, canal 0.
    plot_rp_sample(X_rp, X_rp, sample_index=0, channel=0, save_folder=images_folder)

if __name__ == '__main__':
    main()
