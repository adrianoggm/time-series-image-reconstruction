#!C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/venv/bin/python3
from keras.utils import to_categorical
import argparse
import src.activity_data as act
import src.calculate_ts_errors as cerr
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt
import src.ts_plots as plot
from PIL import Image
from dtaidistance import dtw
import statistics
def generate_latex_tables(results):
    """
    Genera tablas LaTeX para incluir en el README
    """
    print("\n" + "="*80)
    print("TABLAS LaTeX PARA EL README")
    print("="*80)
    
    technique_names = {
        'RP': 'Recurrence Plot',
        'GAF': 'Gramian Angular Field', 
        'MTF': 'Markov Transition Field',
        'GADF': 'Gramian Angular Difference Field'
    }
    
    for technique, data in results.items():
        print(f"\n\\begin{{table}}[H]")
        print("\\centering")
        print(f"\\caption{{{technique_names[technique]} Errors}}")
        print(f"\\label{{tab:{technique.lower()}_errors}}")
        print("\\resizebox{\\textwidth}{!}{%")
        print("\\begin{tabular}{lcccc}")
        print("\\toprule")
        print("\\textbf{Metric} & \\textbf{Dim 0} & \\textbf{Dim 1} & \\textbf{Dim 2} & \\textbf{Global Mean} \\\\")
        print("\\midrule")
        
        # Mapear nombres de métricas
        metric_names = {
            'Error Absoluto Promedio': 'Mean Absolute Error',
            'Error Relativo Promedio': 'Mean Relative Error', 
            'Error Quadrático medio': 'Mean Squared Error',
            'Error desviacion típica': 'Standard Deviation Error',
            'Coeficiente de correlación pearson': 'Pearson Correlation Coefficient'
        }
        
        for metric_key, metric_name in metric_names.items():
            if metric_key in data:
                dim0_mean = data[metric_key]['dim_0']['mean']
                dim0_std = data[metric_key]['dim_0']['std']
                dim1_mean = data[metric_key]['dim_1']['mean']
                dim1_std = data[metric_key]['dim_1']['std']
                dim2_mean = data[metric_key]['dim_2']['mean']
                dim2_std = data[metric_key]['dim_2']['std']
                global_mean = data[metric_key]['global']['mean']
                global_std = data[metric_key]['global']['std']
                
                print(f"{metric_name} & ${dim0_mean:.4f} \\pm {dim0_std:.4f}$ & ${dim1_mean:.4f} \\pm {dim1_std:.4f}$ & ${dim2_mean:.4f} \\pm {dim2_std:.4f}$ & ${global_mean:.4f} \\pm {global_std:.4f}$ \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("}")
        print("\\end{table}")
    
    # Generar tabla de métricas normalizadas basada en nuestros datos
    print(f"\n\\begin{{table}}[H]")
    print("\\centering")
    print("\\caption{Normalized Reconstruction Metrics Comparison}")
    print("\\label{tab:normalized_metrics}")
    print("\\resizebox{0.9\\textwidth}{!}{%")
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("\\textbf{Metric} & \\textbf{GAF} & \\textbf{GADF} & \\textbf{RP} & \\textbf{MTF} & \\textbf{Cubic Spline} \\\\")
    print("\\midrule")
    print("Mean Squared Error & 0.0000 & 0.0210 & 0.1510 & 0.1441 & 0.0000 \\\\")
    print("Mean Absolute Error & 0.0016 & 0.1310 & 0.3212 & 0.3074 & 0.0000 \\\\")
    print("Root Mean Squared Error & 0.0032 & 0.1435 & 0.3833 & 0.3765 & 0.0000 \\\\")
    print("Mean Absolute Percentage Error & 1.50\\% & 33.96\\% & 33,351,350\\% & 23,437,610\\% & 0.00\\% \\\\")
    print("Mean Correlation Coefficient & 0.9999 & 0.9793 & 0.1063 & 0.3079 & 1.0000 \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\end{table}")

def main():
    data_name="WISDM"
    data_folder="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM//tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
    
    # Lista de técnicas a procesar
    techniques = ["GAF", "GADF", "RP", "MTF"]
    
    results = {}
    
    for img_type in techniques:
        print(f"\n=== PROCESANDO {img_type} ===")
        indices_T=np.load(f"{data_folder}/training_data.npy")
        
        if img_type=="RP":
            data_F1="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/recurrence_plot/sampling_loto/3-fold/fold-0/train"
            errores = np.load(f"{data_F1}/errores_rec.npy")
            X_all_rec=np.load(f"{data_folder}/X_all_rec.npy")
        elif img_type=="GAF":
            data_F1="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train"
            errores = np.load(f"{data_F1}/errores_gaf.npy")
            X_all_rec=np.load(f"C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GAF/sampling_loto/3-fold/fold-0/train/X_all_gaf.npy")
        elif img_type=="MTF":
            data_F1="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train"
            errores = np.load(f"{data_F1}/errores_mtf.npy")
            X_all_rec=np.load(f"C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/MTF/sampling_loto/3-fold/fold-0/train/X_all_mtf.npy")
        elif img_type=="GADF":
            data_F1="C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GADF/sampling_loto/3-fold/fold-0/train"
            errores = np.load(f"{data_F1}/errores_gadf.npy")
            X_all_rec=np.load(f"C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/tseries/GADF/sampling_loto/3-fold/fold-0/train/X_all_gadf.npy")
        
        tipoerrores=["Error Absoluto Promedio","Error Relativo Promedio","Error Quadrático medio","Error desviacion típica","Coeficiente de correlación pearson"]
        
        # Almacenar resultados para cada técnica
        technique_results = {}
        
        print("errores",errores.shape)
        for i in range(0,len(tipoerrores)):
            a=tipoerrores[i]
            technique_results[a] = {}
            
            # Por dimensión
            for j in range(0,3):
                mean_val = np.mean(errores[:,i,j][errores[:,i,j]<=1000])
                std_val = np.std(errores[:,i,j][errores[:,i,j]<=1000])
                technique_results[a][f'dim_{j}'] = {'mean': mean_val, 'std': std_val}
                print(f"ERROR TIPO {a} en dim {j} es : {mean_val:.6f} ± {std_val:.6f}")
            
            # Global
            global_mean = np.mean(errores[:,i,:][errores[:,i,:]<=1000])
            global_std = np.std(errores[:,i,:][errores[:,i,:]<=1000])
            technique_results[a]['global'] = {'mean': global_mean, 'std': global_std}
            print(f"ERROR TIPO {a} medio global es : {global_mean:.6f} ± {global_std:.6f}")
        
        results[img_type] = technique_results
    
    # Generar tablas LaTeX
    generate_latex_tables(results)
    
   
     
# Guardar el gráfico como una imagen
if __name__ == '__main__':
    main()
