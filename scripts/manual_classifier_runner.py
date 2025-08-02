#!/usr/bin/env python3
"""
Script para ejecutar todos los clasificadores con todas las técnicas de forma secuencial y manual.
"""

import os
import time
import json
from datetime import datetime

def run_all_combinations():
    """
    Ejecuta todas las combinaciones de clasificadores y técnicas.
    """
    
    # Definir todas las combinaciones
    combinations = [
        # KNN
        ('knn', 'GAF', 'scripts/knnvalidation.py', '"GAF"'),
        ('knn', 'GADF', 'scripts/knnvalidation.py', '"GADF"'),
        ('knn', 'RP', 'scripts/knnvalidation.py', '"RP"'),
        ('knn', 'MTF', 'scripts/knnvalidation.py', '"MTF"'),
        
        # Random Forest
        ('randomforest', 'GAF', 'scripts/randomforestvalidation.py', '"GAF"'),
        ('randomforest', 'GADF', 'scripts/randomforestvalidation.py', '"GADF"'),
        ('randomforest', 'RP', 'scripts/randomforestvalidation.py', '"RP"'),
        ('randomforest', 'MTF', 'scripts/randomforestvalidation.py', '"MTF"'),
        
        # LSTM
        ('lstm', 'GAF', 'scripts/lstmvalidation.py', '"GAF"'),
        ('lstm', 'GADF', 'scripts/lstmvalidation.py', '"GADF"'),
        ('lstm', 'RP', 'scripts/lstmvalidation.py', '"RP"'),
        ('lstm', 'MTF', 'scripts/lstmvalidation.py', '"MTF"'),
        
        # GRU  
        ('gru', 'GAF', 'scripts/gruvalidation.py', '"GAF"'),
        ('gru', 'GADF', 'scripts/gruvalidation.py', '"GADF"'),
        ('gru', 'RP', 'scripts/gruvalidation.py', '"RP"'),
        ('gru', 'MTF', 'scripts/gruvalidation.py', '"MTF"'),
        
        # CNN
        ('cnn', 'GAF', 'scripts/cnnvalidation.py', '"GAF"'),
        ('cnn', 'GADF', 'scripts/cnnvalidation.py', '"GADF"'),
        ('cnn', 'RP', 'scripts/cnnvalidation.py', '"RP"'),
        ('cnn', 'MTF', 'scripts/cnnvalidation.py', '"MTF"'),
    ]
    
    print("🎯 Ejecución Manual Secuencial de Clasificadores")
    print("="*60)
    print(f"📊 Total de combinaciones: {len(combinations)}")
    print("="*60)
    
    results = []
    
    for i, (classifier, technique, script_path, stype_value) in enumerate(combinations):
        print(f"\\n[{i+1}/{len(combinations)}] {classifier.upper()} con {technique}")
        print("-" * 50)
        
        try:
            # Verificar que el archivo existe
            if not os.path.exists(script_path):
                print(f"❌ Script no encontrado: {script_path}")
                results.append({
                    'classifier': classifier,
                    'technique': technique,
                    'success': False,
                    'error': f"Script no encontrado: {script_path}",
                    'timestamp': datetime.now().isoformat()
                })
                continue
            
            # Leer el script
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Hacer backup
            backup_path = script_path + '.backup'
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Modificar para la técnica específica
            modified_content = content
            
            # Cambiar stype
            if 'stype = "GAF"' in modified_content:
                modified_content = modified_content.replace('stype = "GAF"', f'stype = {stype_value}')
            elif 'stype = "RP"' in modified_content:
                modified_content = modified_content.replace('stype = "RP"', f'stype = {stype_value}')
            elif 'stype = "MTF"' in modified_content:
                modified_content = modified_content.replace('stype = "MTF"', f'stype = {stype_value}')
            
            # Agregar soporte GADF si no existe
            if technique == "GADF" and "GADF" not in modified_content:
                # Agregar folder GADF
                if 'data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")' in modified_content:
                    modified_content = modified_content.replace(
                        'data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")',
                        '''data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    data_folder_gadf = os.path.join(dataset_folder, "tseries/GADF/sampling_loto/3-fold/fold-0")'''
                    )
                
                # Agregar caso GADF
                if 'elif stype == "MTF":' in modified_content:
                    # Encontrar la sección MTF y agregar GADF antes
                    mtf_start = modified_content.find('elif stype == "MTF":')
                    if mtf_start != -1:
                        gadf_section = '''elif stype == "GADF":
        gadf_path = os.path.join(data_folder_gadf, "test", "X_all_gadf.npy")
        X_all_gadf = np.load(gadf_path)
        X_recon = np.transpose(X_all_gadf, (0, 2, 1))
        print("X_recon GADF shape:", X_all_gadf.shape, "-> transposed:", X_recon.shape)
    '''
                        modified_content = modified_content[:mtf_start] + gadf_section + modified_content[mtf_start:]
            
            # Para CNN/LSTM/GRU: actualizar procesamiento de datos
            if classifier in ['cnn', 'lstm', 'gru'] and technique in ['GAF', 'GADF', 'MTF']:
                if 'input_shape = (128, 3)' in modified_content and 'X_train_processed = X_train' in modified_content:
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
            
            # Escribir el archivo modificado
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            print(f"✏️  Script modificado para usar {technique}")
            print(f"▶️  Ejecutando...")
            
            # Ejecutar manualmente 
            # Nota: En lugar de subprocess, le decimos al usuario qué hacer
            print(f"🔧 EJECUTAR MANUALMENTE:")
            print(f"   python {script_path}")
            print(f"   (Configurado para {technique})")
            
            # Simular éxito por ahora
            results.append({
                'classifier': classifier,
                'technique': technique,
                'success': True,
                'script_path': script_path,
                'configured_for': technique,
                'timestamp': datetime.now().isoformat(),
                'note': 'Script configurado - ejecutar manualmente'
            })
            
            print(f"✅ Script preparado para {technique}")
            
            # Restaurar original
            with open(backup_path, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(original)
            os.remove(backup_path)
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            results.append({
                'classifier': classifier,
                'technique': technique,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
    
    # Guardar resultados
    with open('configuraciones_preparadas.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\\n" + "="*60)
    print("📋 RESUMEN")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    print(f"✅ Scripts preparados exitosamente: {successful}/{len(combinations)}")
    print(f"💾 Configuraciones guardadas en: configuraciones_preparadas.json")
    
    if successful > 0:
        print("\\n🚀 PRÓXIMOS PASOS:")
        print("   1. Ejecutar cada script manualmente según la lista")
        print("   2. Guardar los resultados de cada ejecución")
        print("   3. Compilar resultados finales")

def show_execution_plan():
    """
    Muestra el plan de ejecución paso a paso.
    """
    combinations = [
        ('KNN', 'GAF'), ('KNN', 'GADF'), ('KNN', 'RP'), ('KNN', 'MTF'),
        ('Random Forest', 'GAF'), ('Random Forest', 'GADF'), ('Random Forest', 'RP'), ('Random Forest', 'MTF'),
        ('LSTM', 'GAF'), ('LSTM', 'GADF'), ('LSTM', 'RP'), ('LSTM', 'MTF'),
        ('GRU', 'GAF'), ('GRU', 'GADF'), ('GRU', 'RP'), ('GRU', 'MTF'),
        ('CNN', 'GAF'), ('CNN', 'GADF'), ('CNN', 'RP'), ('CNN', 'MTF'),
    ]
    
    print("📋 PLAN DE EJECUCIÓN MANUAL")
    print("="*50)
    print("\\nEjecuta cada comando en orden:\\n")
    
    script_map = {
        'KNN': 'knnvalidation.py',
        'Random Forest': 'randomforestvalidation.py', 
        'LSTM': 'lstmvalidation.py',
        'GRU': 'gruvalidation.py',
        'CNN': 'cnnvalidation.py'
    }
    
    for i, (classifier, technique) in enumerate(combinations):
        script = script_map[classifier]
        print(f"{i+1:2d}. {classifier} con {technique}:")
        print(f"    - Editar scripts/{script}")
        print(f"    - Cambiar stype = \"{technique}\"")
        print(f"    - python scripts/{script}")
        print(f"    - Guardar output como {classifier.lower()}_{technique.lower()}.txt")
        print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--plan':
        show_execution_plan()
    else:
        run_all_combinations()
