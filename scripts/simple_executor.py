#!/usr/bin/env python3
"""
Script simple para ejecutar clasificadores uno por uno, evitando problemas de codificación.
"""

import subprocess
import sys
import os
import time
from datetime import datetime

def run_single_combination(classifier, technique):
    """
    Ejecuta una sola combinación de clasificador y técnica.
    """
    script_map = {
        'knn': 'knnvalidation.py',
        'randomforest': 'randomforestvalidation.py',
        'lstm': 'lstmvalidation.py',
        'gru': 'gruvalidation.py',
        'cnn': 'cnnvalidation.py'
    }
    
    if classifier not in script_map:
        return False, f"Clasificador desconocido: {classifier}"
    
    script_name = script_map[classifier]
    script_path = os.path.join("scripts", script_name)
    
    print(f"Ejecutando {classifier.upper()} con {technique}...")
    
    try:
        start_time = time.time()
        
        # Ejecutar el comando desde el directorio raíz del proyecto
        cmd = [sys.executable, script_path, "--technique", technique]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding='utf-8',
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Directorio padre de scripts
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Guardar resultado en archivo
        output_file = f"resultado_{classifier}_{technique.lower()}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {classifier.upper()} con {technique} ===\\n")
            f.write(f"Ejecutado: {datetime.now()}\\n")
            f.write(f"Duracion: {duration:.2f} segundos\\n")
            f.write(f"Comando: {' '.join(cmd)}\\n")
            f.write(f"Codigo de retorno: {result.returncode}\\n")
            f.write("="*50 + "\\n\\n")
            f.write("STDOUT:\\n")
            f.write(result.stdout)
            f.write("\\n\\nSTDERR:\\n")
            f.write(result.stderr)
        
        if result.returncode == 0:
            print(f"  -> EXITO en {duration:.1f}s. Guardado en {output_file}")
            return True, output_file
        else:
            print(f"  -> FALLO en {duration:.1f}s. Ver {output_file}")
            return False, output_file
            
    except Exception as e:
        error_msg = str(e)
        print(f"  -> ERROR: {error_msg}")
        return False, error_msg

def main():
    """
    Ejecuta todas las combinaciones secuencialmente.
    """
    combinations = [
        # Ya ejecutados: KNN con GAF, GADF, RP - Continuamos desde MTF
        ('knn', 'MTF'),
        ('randomforest', 'GAF'), ('randomforest', 'GADF'), ('randomforest', 'RP'), ('randomforest', 'MTF'),
        ('lstm', 'GAF'), ('lstm', 'GADF'), ('lstm', 'RP'), ('lstm', 'MTF'),
        ('gru', 'GAF'), ('gru', 'GADF'), ('gru', 'RP'), ('gru', 'MTF'),
        ('cnn', 'GAF'), ('cnn', 'GADF'), ('cnn', 'RP'), ('cnn', 'MTF'),
    ]
    
    print("EXECUTOR SIMPLE DE CLASIFICADORES")
    print("="*50)
    print(f"Total de combinaciones: {len(combinations)}")
    print("="*50)
    
    # Verificar datos GADF
    data_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
    gadf_test_path = os.path.join(data_folder, "tseries/GADF/sampling_loto/3-fold/fold-0/test/X_all_gadf.npy")
    
    if not os.path.exists(gadf_test_path):
        print("ADVERTENCIA: Datos GADF no encontrados. Se omitiran combinaciones GADF.")
        combinations = [(c, t) for c, t in combinations if t != 'GADF']
        print(f"Combinaciones actualizadas: {len(combinations)}")
    
    results = []
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, (classifier, technique) in enumerate(combinations):
        print(f"\\n[{i+1}/{len(combinations)}] {classifier.upper()} + {technique}")
        print("-" * 40)
        
        success, info = run_single_combination(classifier, technique)
        
        results.append({
            'classifier': classifier,
            'technique': technique,
            'success': success,
            'info': info,
            'timestamp': datetime.now().isoformat()
        })
        
        if success:
            successful += 1
        else:
            failed += 1
        
        print(f"Estado actual: {successful} exitosos, {failed} fallidos")
    
    # Resumen final
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\\n" + "="*50)
    print("RESUMEN FINAL")
    print("="*50)
    print(f"Exitosos: {successful}")
    print(f"Fallidos: {failed}")
    print(f"Total: {len(combinations)}")
    print(f"Tasa de exito: {successful/len(combinations)*100:.1f}%")
    print(f"Duracion total: {total_duration/60:.1f} minutos")
    
    # Guardar resumen
    import json
    with open('resumen_ejecucion_simple.json', 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'duration_minutes': total_duration / 60,
            'successful': successful,
            'failed': failed,
            'total': len(combinations),
            'success_rate': successful / len(combinations) * 100,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print("\\nResumen guardado en: resumen_ejecucion_simple.json")
    print("Archivos individuales: resultado_<clasificador>_<tecnica>.txt")

if __name__ == "__main__":
    main()
