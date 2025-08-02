#!/usr/bin/env python3
"""
Script para generar gráficas comparativas entre datos originales y reconstruidos
usando diferentes técnicas (GAF, GADF, RP, MTF) y calcular interpolación con spline cúbico.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Configuración de estilo para las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 11

# Nombres de los canales
CHANNEL_NAMES = ['X-axis', 'Y-axis', 'Z-axis']
CHANNEL_COLORS_ORIG = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Azul, naranja, verde
CHANNEL_COLORS_REC = ['#d62728', '#9467bd', '#8c564b']   # Rojo, púrpura, marrón

def normalize_time_series(data, method='minmax'):
    """
    Normaliza series temporales usando diferentes métodos.
    
    Args:
        data: Array de forma (n_samples, time_steps, n_channels) o (time_steps, n_channels)
        method: Método de normalización ('minmax', 'zscore')
    
    Returns:
        normalized_data: Datos normalizados con la misma forma
        scalers: Lista de scalers usados (para desnormalizar si es necesario)
    """
    original_shape = data.shape
    
    if len(original_shape) == 2:
        # Caso (time_steps, n_channels)
        if method == 'minmax':
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(data)
            return normalized, [scaler]
        elif method == 'zscore':
            # Z-score normalización por canal
            normalized = np.zeros_like(data)
            scalers = []
            for channel in range(data.shape[1]):
                mean = np.mean(data[:, channel])
                std = np.std(data[:, channel])
                normalized[:, channel] = (data[:, channel] - mean) / std
                scalers.append({'mean': mean, 'std': std})
            return normalized, scalers
    
    elif len(original_shape) == 3:
        # Caso (n_samples, time_steps, n_channels)
        normalized = np.zeros_like(data)
        scalers = []
        
        for sample in range(original_shape[0]):
            if method == 'minmax':
                scaler = MinMaxScaler()
                normalized[sample] = scaler.fit_transform(data[sample])
                scalers.append(scaler)
            elif method == 'zscore':
                sample_scalers = []
                for channel in range(original_shape[2]):
                    mean = np.mean(data[sample, :, channel])
                    std = np.std(data[sample, :, channel])
                    normalized[sample, :, channel] = (data[sample, :, channel] - mean) / std
                    sample_scalers.append({'mean': mean, 'std': std})
                scalers.append(sample_scalers)
        
        return normalized, scalers
    
    return data, []

def calculate_cubic_spline_interpolation(original_data, n_points=256):
    """
    Calcula la interpolación con spline cúbico de los datos originales.
    
    Args:
        original_data: Array de forma (time_steps, channels)
        n_points: Número de puntos para la interpolación
    
    Returns:
        interpolated_data: Datos interpolados
        x_new: Nuevos puntos temporales
    """
    time_steps, channels = original_data.shape
    x_original = np.linspace(0, 1, time_steps)
    x_new = np.linspace(0, 1, n_points)
    
    interpolated_data = np.zeros((n_points, channels))
    
    for channel in range(channels):
        cs = CubicSpline(x_original, original_data[:, channel])
        interpolated_data[:, channel] = cs(x_new)
    
    return interpolated_data, x_new

def generate_cubic_spline_dataset(original_data, target_points=128):
    """
    Genera un dataset completo de interpolaciones cúbicas.
    
    Args:
        original_data: Array de forma (n_samples, time_steps, channels)
        target_points: Número de puntos temporales deseados
    
    Returns:
        spline_data: Array de forma (n_samples, target_points, channels)
    """
    n_samples, time_steps, channels = original_data.shape
    spline_data = np.zeros((n_samples, target_points, channels))
    
    for sample_idx in range(n_samples):
        interpolated, _ = calculate_cubic_spline_interpolation(
            original_data[sample_idx], 
            n_points=target_points
        )
        spline_data[sample_idx] = interpolated
    
    return spline_data

def calculate_reconstruction_metrics(original, reconstructed, normalize=True):
    """
    Calcula métricas de error entre datos originales y reconstruidos.
    
    Args:
        original: Serie temporal original (time_steps, channels)
        reconstructed: Serie temporal reconstruida (time_steps, channels)
        normalize: Si True, normaliza los datos antes de calcular métricas
    """
    # Verificar que las dimensiones sean compatibles
    if original.shape != reconstructed.shape:
        print(f"⚠️  Dimensiones incompatibles: original {original.shape} vs reconstructed {reconstructed.shape}")
        return {
            'MSE': float('nan'),
            'MAE': float('nan'),
            'RMSE': float('nan'),
            'MAPE': float('nan'),
            'Correlations': [float('nan')] * original.shape[1],
            'Mean_Correlation': float('nan')
        }
    
    # Normalizar datos si se especifica
    if normalize:
        orig_norm, _ = normalize_time_series(original, method='minmax')
        rec_norm, _ = normalize_time_series(reconstructed, method='minmax')
    else:
        orig_norm = original
        rec_norm = reconstructed
    
    # Calcular métricas básicas
    mse = mean_squared_error(orig_norm.flatten(), rec_norm.flatten())
    mae = mean_absolute_error(orig_norm.flatten(), rec_norm.flatten())
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error) - solo para valores no zero
    mape = np.mean(np.abs((orig_norm - rec_norm) / (orig_norm + 1e-8))) * 100
    
    # Correlación por canal
    correlations = []
    for channel in range(original.shape[1]):
        # Usar datos originales sin normalizar para correlación
        corr = np.corrcoef(original[:, channel], reconstructed[:, channel])[0, 1]
        if np.isnan(corr):
            corr = 0.0  # Si hay NaN, asumir correlación 0
        correlations.append(corr)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Correlations': correlations,
        'Mean_Correlation': np.mean(correlations)
    }

def plot_global_comparison(original, reconstructed, sample_idx, technique, save_path):
    """
    Genera gráfica comparativa global (todos los canales).
    """
    time_original = np.linspace(0, 1, original.shape[1])
    time_reconstructed = np.linspace(0, 1, reconstructed.shape[1])
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plotear datos originales
    for channel in range(3):
        ax.plot(time_original, original[sample_idx, :, channel], 
                label=f'Original {CHANNEL_NAMES[channel]}', 
                color=CHANNEL_COLORS_ORIG[channel], 
                linewidth=2, alpha=0.8)
    
    # Plotear datos reconstruidos
    for channel in range(3):
        ax.plot(time_reconstructed, reconstructed[sample_idx, :, channel], 
                label=f'{technique} {CHANNEL_NAMES[channel]}', 
                color=CHANNEL_COLORS_REC[channel], 
                linewidth=2, alpha=0.8, linestyle='--')
    
    ax.set_title(f'Comparación Global: Original vs {technique} - Muestra {sample_idx}', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Tiempo Normalizado', fontsize=14)
    ax.set_ylabel('Valor de Aceleración', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{technique.lower()}_global_comparison_sample_{sample_idx}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_channel_comparison(original, reconstructed, sample_idx, channel, technique, save_path):
    """
    Genera gráfica comparativa por canal individual.
    """
    time_original = np.linspace(0, 1, original.shape[1])
    time_reconstructed = np.linspace(0, 1, reconstructed.shape[1])
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
    
    # Gráfica original
    ax1.plot(time_original, original[sample_idx, :, channel], 
             color=CHANNEL_COLORS_ORIG[channel], linewidth=2)
    ax1.set_title(f'Original - {CHANNEL_NAMES[channel]} - Muestra {sample_idx}', 
                  fontweight='bold')
    ax1.set_ylabel('Valor')
    ax1.grid(True, alpha=0.3)
    
    # Gráfica reconstruida
    ax2.plot(time_reconstructed, reconstructed[sample_idx, :, channel], 
             color=CHANNEL_COLORS_REC[channel], linewidth=2)
    ax2.set_title(f'{technique} Reconstruido - {CHANNEL_NAMES[channel]} - Muestra {sample_idx}', 
                  fontweight='bold')
    ax2.set_ylabel('Valor')
    ax2.grid(True, alpha=0.3)
    
    # Comparación superpuesta
    ax3.plot(time_original, original[sample_idx, :, channel], 
             label='Original', color=CHANNEL_COLORS_ORIG[channel], linewidth=2)
    ax3.plot(time_reconstructed, reconstructed[sample_idx, :, channel], 
             label=f'{technique}', color=CHANNEL_COLORS_REC[channel], 
             linewidth=2, linestyle='--', alpha=0.8)
    ax3.set_title(f'Comparación Superpuesta - {CHANNEL_NAMES[channel]} - Muestra {sample_idx}', 
                  fontweight='bold')
    ax3.set_xlabel('Tiempo Normalizado')
    ax3.set_ylabel('Valor')
    ax3.legend()  
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'{technique.lower()}_channel_{channel}_comparison_sample_{sample_idx}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename

def plot_cubic_spline_comparison(original, sample_idx, save_path):
    """
    Genera gráfica mostrando interpolación con spline cúbico.
    """
    # Calcular spline cúbico
    interpolated, x_new = calculate_cubic_spline_interpolation(original[sample_idx])
    time_original = np.linspace(0, 1, original.shape[1])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Comparación global
    ax = axes[0]
    for channel in range(3):
        ax.plot(time_original, original[sample_idx, :, channel], 
                'o-', label=f'Original {CHANNEL_NAMES[channel]}', 
                color=CHANNEL_COLORS_ORIG[channel], alpha=0.7, markersize=4)
        ax.plot(x_new, interpolated[:, channel], 
                '--', label=f'Spline {CHANNEL_NAMES[channel]}', 
                color=CHANNEL_COLORS_REC[channel], linewidth=2)
    ax.set_title('Interpolación Spline Cúbico - Global', fontweight='bold')
    ax.set_xlabel('Tiempo Normalizado')
    ax.set_ylabel('Valor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Comparaciones por canal
    for channel in range(3):
        ax = axes[channel + 1]
        ax.plot(time_original, original[sample_idx, :, channel], 
                'o-', label='Original', color=CHANNEL_COLORS_ORIG[channel], 
                alpha=0.7, markersize=4)
        ax.plot(x_new, interpolated[:, channel], 
                '--', label='Spline Cúbico', color=CHANNEL_COLORS_REC[channel], 
                linewidth=2)
        ax.set_title(f'Spline Cúbico - {CHANNEL_NAMES[channel]}', fontweight='bold')
        ax.set_xlabel('Tiempo Normalizado')
        ax.set_ylabel('Valor')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'cubic_spline_comparison_sample_{sample_idx}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    return filename, interpolated

def generate_metrics_report(original_data, reconstructed_data, techniques, save_path):
    """
    Genera un reporte con métricas de reconstrucción para todas las técnicas.
    """
    results = []
    
    for i, technique in enumerate(techniques):
        if reconstructed_data[i] is not None:
            # Calcular métricas para las primeras 10 muestras
            for sample_idx in range(min(10, original_data.shape[0])):
                metrics = calculate_reconstruction_metrics(
                    original_data[sample_idx], 
                    reconstructed_data[i][sample_idx]
                )
                
                result = {
                    'Technique': technique,
                    'Sample': sample_idx,
                    'MSE': metrics['MSE'],
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'MAPE': metrics['MAPE'],
                    'Mean_Correlation': metrics['Mean_Correlation'],
                    'Corr_X': metrics['Correlations'][0],
                    'Corr_Y': metrics['Correlations'][1],
                    'Corr_Z': metrics['Correlations'][2]
                }
                results.append(result)
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(save_path, 'reconstruction_metrics.csv'), index=False)
    
    # Generar resumen estadístico
    summary = df.groupby('Technique').agg({
        'MSE': ['mean', 'std'],
        'MAE': ['mean', 'std'], 
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std'],
        'Mean_Correlation': ['mean', 'std']
    }).round(4)
    
    summary.to_csv(os.path.join(save_path, 'reconstruction_metrics_summary.csv'))
    
    return df, summary

def main():
    parser = argparse.ArgumentParser(description='Generar gráficas comparativas y cálculos de spline cúbico')
    parser.add_argument('--sample-indices', nargs='+', type=int, default=[0, 1, 2], 
                       help='Índices de muestras a analizar')
    parser.add_argument('--output-dir', type=str, default='images', 
                       help='Directorio de salida para las gráficas')
    
    args = parser.parse_args()
    
    # Crear directorio de salida
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generando gráficas en: {output_dir}")
    
    # Rutas de datos
    dataset_folder = "data/WISDM"
    data_folder_base = os.path.join(dataset_folder, "tseries")
    
    # Cargar datos originales
    print("Cargando datos originales...")
    train_path = os.path.join(data_folder_base, "recurrence_plot/sampling_loto/3-fold/fold-0/train/training_data.npy")
    original_data = np.load(train_path)
    print(f"Datos originales cargados: {original_data.shape}")
    
    # Rutas de datos reconstruidos
    techniques = ['GAF', 'GADF', 'RP', 'MTF', 'Cubic_Spline']
    reconstructed_paths = {
        'GAF': os.path.join(data_folder_base, "GAF/sampling_loto/3-fold/fold-0/train/X_all_gaf.npy"),
        'GADF': os.path.join(data_folder_base, "GADF/sampling_loto/3-fold/fold-0/train/X_all_gadf.npy"),
        'RP': os.path.join(data_folder_base, "recurrence_plot/sampling_loto/3-fold/fold-0/train/X_all_rec.npy"),
        'MTF': os.path.join(data_folder_base, "MTF/sampling_loto/3-fold/fold-0/train/X_all_mtf.npy")
    }
    
    # Cargar datos reconstruidos
    reconstructed_data = []
    for technique in techniques:
        if technique == 'Cubic_Spline':
            # Generar interpolaciones cúbicas
            print(f"Generando datos {technique}...")
            spline_data = generate_cubic_spline_dataset(original_data, target_points=128)
            reconstructed_data.append(spline_data)
            print(f"Datos {technique} generados: {spline_data.shape}")
        else:
            path = reconstructed_paths[technique]
            if os.path.exists(path):
                print(f"Cargando datos {technique}...")
                data = np.load(path)
                # Transponer si es necesario (de (n, channels, time) a (n, time, channels))
                if technique in ['GAF', 'GADF', 'RP', 'MTF']:
                    data = np.transpose(data, (0, 2, 1))
                reconstructed_data.append(data)
                print(f"Datos {technique} cargados: {data.shape}")
            else:
                print(f"⚠️  Archivo no encontrado: {path}")
                reconstructed_data.append(None)
    
    print("\nGenerando gráficas comparativas...")
    
    # Generar gráficas para las muestras especificadas
    for sample_idx in args.sample_indices:
        if sample_idx >= original_data.shape[0]:
            print(f"⚠️  Muestra {sample_idx} no disponible (máximo: {original_data.shape[0]-1})")
            continue
            
        print(f"\nProcesando muestra {sample_idx}...")
        
        # Generar gráfica de spline cúbico
        print("  - Generando gráfica de spline cúbico...")
        spline_file, interpolated = plot_cubic_spline_comparison(original_data, sample_idx, output_dir)
        print(f"    ✓ {spline_file}")
        
        # Generar gráficas comparativas para cada técnica
        for i, technique in enumerate(techniques):
            if reconstructed_data[i] is not None:
                print(f"  - Generando gráficas para {technique}...")
                
                # Gráfica global
                global_file = plot_global_comparison(
                    original_data, reconstructed_data[i], sample_idx, technique, output_dir
                )
                print(f"    ✓ {global_file}")
                
                # Gráficas por canal
                for channel in range(3):
                    channel_file = plot_channel_comparison(
                        original_data, reconstructed_data[i], sample_idx, channel, technique, output_dir
                    )
                    print(f"    ✓ {channel_file}")
    
    # Generar reporte de métricas
    print("\nGenerando reporte de métricas...")
    try:
        df_metrics, summary_metrics = generate_metrics_report(
            original_data, reconstructed_data, techniques, output_dir
        )
        print("✓ reconstruction_metrics.csv")
        print("✓ reconstruction_metrics_summary.csv")
        
        # Mostrar resumen
        print("\n" + "="*60)
        print("RESUMEN DE MÉTRICAS DE RECONSTRUCCIÓN")
        print("="*60)
        print(summary_metrics)
        
    except Exception as e:
        print(f"⚠️  Error generando métricas: {e}")
    
    print(f"\n🎉 Proceso completado. Gráficas guardadas en: {output_dir}")
    print("\nArchivos generados:")
    for file in sorted(os.listdir(output_dir)):
        if file.endswith('.png') or file.endswith('.csv'):
            print(f"  - {file}")

if __name__ == '__main__':
    main()
