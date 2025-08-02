#!/usr/bin/env python3
"""
Script para ejecutar todos los clasificadores con todas las técnicas de manera secuencial.
Este script manualmente invoca cada clasificador modificando temporalmente el código.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

def run_classifier_evaluation(classifier_name, technique, results_file):
    """
    Ejecuta evaluación para un clasificador específico con una técnica específica.
    Modifica temporalmente el script y lo ejecuta.
    """
    
    # Mapeo de scripts
    script_files = {
        'cnn': 'cnnvalidation.py',
        'lstm': 'lstmvalidation.py',
        'gru': 'gruvalidation.py', 
        'knn': 'knnvalidation.py',
        'randomforest': 'randomforestvalidation.py'
    }
    
    if classifier_name not in script_files:
        return False, f"Clasificador {classifier_name} no encontrado"
    
    script_name = script_files[classifier_name]
    script_path = os.path.join("scripts", script_name)
    
    if not os.path.exists(script_path):
        return False, f"Script {script_path} no encontrado"
    
    print(f"\\n🚀 Ejecutando {classifier_name.upper()} con {technique}")
    print("-" * 50)
    
    try:
        # Leer el script original
        with open(script_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Crear backup temporal
        backup_path = script_path + '.backup'
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Modificar el contenido para usar la técnica específica
        modified_content = original_content.replace(
            'stype = "GAF"', 
            f'stype = "{technique}"'
        )
        
        # También manejar otros posibles valores por defecto
        modified_content = modified_content.replace(
            'stype = "RP"', 
            f'stype = "{technique}"'
        )
        modified_content = modified_content.replace(
            'stype = "MTF"', 
            f'stype = "{technique}"'
        )
        
        # Actualizar la sección de carga de datos para incluir GADF
        if technique == "GADF" and "GADF" not in modified_content:
            # Buscar la sección donde se cargan los datos reconstruidos
            if 'data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")' in modified_content:
                modified_content = modified_content.replace(
                    'data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")',
                    '''data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    data_folder_gadf = os.path.join(dataset_folder, "tseries/GADF/sampling_loto/3-fold/fold-0")'''
                )
            
            # Agregar el caso GADF en el if/elif
            if 'elif stype == "MTF":' in modified_content:
                mtf_section = '''elif stype == "MTF":
        mtf_path = os.path.join(data_folder_mtf, "test", "X_all_mtf.npy")
        X_all_mtf = np.load(mtf_path)
        X_recon = np.transpose(X_all_mtf, (0, 2, 1))'''
                
                gadf_section = '''elif stype == "GADF":
        gadf_path = os.path.join(data_folder_gadf, "test", "X_all_gadf.npy")
        X_all_gadf = np.load(gadf_path)
        X_recon = np.transpose(X_all_gadf, (0, 2, 1))
        print("X_recon GADF shape:", X_all_gadf.shape, "-> transposed:", X_recon.shape)
    elif stype == "MTF":
        mtf_path = os.path.join(data_folder_mtf, "test", "X_all_mtf.npy")
        X_all_mtf = np.load(mtf_path)
        X_recon = np.transpose(X_all_mtf, (0, 2, 1))'''
                
                modified_content = modified_content.replace(mtf_section, gadf_section)
        
        # Para CNN, LSTM, GRU: Asegurar que GADF use el procesamiento correcto
        if classifier_name in ['cnn', 'lstm', 'gru'] and technique in ['GAF', 'GADF', 'MTF']:
            if 'input_shape = (128, 3)' in modified_content and 'X_train_processed = X_train' in modified_content:
                # Actualizar el procesamiento para técnicas que no son RP
                modified_content = modified_content.replace(
                    '''    # Transformación de datos según la técnica.
    # Para "RP": se descarta el último time step (de 129 se toman 128)
    X_train_processed = X_train
    X_val_processed = X_val
    input_shape = (128, 3)''',
                    '''    # Transformación de datos según la técnica.
    if stype == "RP":
        # Para RP: se descarta el último time step (de 129 se toman 128)
        X_train_processed = X_train[:, :-1, :]
        X_val_processed = X_val[:, :-1, :]
        input_shape = (128, 3)
    else:
        # Para GAF, GADF, MTF: se asume la forma (129, 3)
        X_train_processed = X_train
        X_val_processed = X_val
        input_shape = (129, 3)'''
                )
        
        # Escribir el contenido modificado
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        # Ejecutar el script
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Directorio raíz del proyecto
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Restaurar el script original
        with open(backup_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        os.remove(backup_path)
        
        # Guardar resultados
        result_data = {
            'classifier': classifier_name,
            'technique': technique,
            'timestamp': datetime.now().isoformat(),
            'duration': duration,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'return_code': result.returncode
        }
        
        # Agregar al archivo de resultados
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                all_results = json.load(f)
        else:
            all_results = []
        
        all_results.append(result_data)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        if result.returncode == 0:
            print(f"✅ Completado exitosamente en {duration:.1f}s")
            print("📊 Últimas líneas de salida:")
            output_lines = result.stdout.split('\\n')[-10:]
            for line in output_lines:
                if line.strip():
                    print(f"   {line}")
            return True, "Éxito"
        else:
            print(f"❌ Falló en {duration:.1f}s")
            print("📋 Error:")
            print(result.stderr[:500])  # Primeros 500 caracteres del error
            return False, result.stderr
            
    except Exception as e:
        # Asegurar que se restaure el script original en caso de error
        try:
            if os.path.exists(backup_path):
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()
                with open(script_path, 'w', encoding='utf-8') as f:
                    f.write(original_content)
                os.remove(backup_path)
        except:
            pass
        
        return False, str(e)

def main():
    """
    Ejecuta todos los clasificadores con todas las técnicas.
    """
    print("🎯 Ejecutor de Clasificadores con Técnicas de Reconstrucción")
    print("="*65)
    
    # Verificar que tenemos los datos de GADF
    data_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
    gadf_test_path = os.path.join(data_folder, "tseries/GADF/sampling_loto/3-fold/fold-0/test/X_all_gadf.npy")
    
    classifiers = ['knn', 'randomforest', 'lstm', 'gru', 'cnn']  # Orden por velocidad
    techniques = ['GAF', 'RP', 'MTF']
    
    if os.path.exists(gadf_test_path):
        techniques.append('GADF')
        print("✅ Datos GADF encontrados - se incluirá en la evaluación")
    else:
        print("⚠️  Datos GADF no encontrados - se omitirá GADF")
    
    total_combinations = len(classifiers) * len(techniques)
    results_file = "resultados_clasificadores.json"
    
    print(f"\\n📊 Ejecutando {total_combinations} combinaciones:")
    print(f"   Clasificadores: {classifiers}")
    print(f"   Técnicas: {techniques}")
    print(f"   Resultados en: {results_file}")
    print("="*65)
    
    start_time = time.time()
    successful = 0
    failed = 0
    current = 0
    
    for classifier in classifiers:
        for technique in techniques:
            current += 1
            print(f"\\n[{current}/{total_combinations}] {classifier.upper()} + {technique}")
            
            success, message = run_classifier_evaluation(classifier, technique, results_file)
            
            if success:
                successful += 1
                print(f"✅ Éxito")
            else:
                failed += 1
                print(f"❌ Falló: {message[:100]}...")
    
    # Resumen final
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\\n" + "="*65)
    print("📋 RESUMEN FINAL")
    print("="*65)
    print(f"   ✅ Exitosos: {successful}")
    print(f"   ❌ Fallidos: {failed}")
    print(f"   📊 Total: {total_combinations}")
    print(f"   🎯 Tasa de éxito: {successful/total_combinations*100:.1f}%")
    print(f"   ⏱️  Duración total: {total_duration/60:.1f} minutos")
    print(f"   💾 Resultados guardados en: {results_file}")
    
    if successful == total_combinations:
        print("\\n🎉 ¡Todas las evaluaciones completadas exitosamente!")
    else:
        print(f"\\n⚠️  {failed} evaluaciones fallaron. Revisa el archivo de resultados.")

if __name__ == "__main__":
    main()
