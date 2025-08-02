#!/usr/bin/env python3
"""
Script para preparar y ejecutar todos los clasificadores con todas las técnicas de reconstrucción.
Incluye modificaciones automáticas de los scripts para soportar GADF.
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime

def modify_script_for_gadf_support(script_path):
    """
    Modifica un script de validación para incluir soporte completo para GADF.
    """
    if not os.path.exists(script_path):
        print(f"⚠️  Script no encontrado: {script_path}")
        return False
        
    try:
        # Leer el contenido actual
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 1. Agregar import argparse si no existe
        if 'import argparse' not in content:
            import_lines = []
            other_lines = []
            in_imports = True
            
            for line in content.split('\\n'):
                if in_imports and (line.startswith('import ') or line.startswith('from ') or line.strip() == '' or line.startswith('#')):
                    import_lines.append(line)
                else:
                    if in_imports:
                        import_lines.append('import argparse')
                        in_imports = False
                    other_lines.append(line)
            
            content = '\\n'.join(import_lines + other_lines)
        
        # 2. Agregar función parse_arguments si no existe
        if 'def parse_arguments():' not in content:
            parser_function = '''
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run classifier validation')
    parser.add_argument('--technique', type=str, default='GAF',
                       choices=['GAF', 'GADF', 'RP', 'MTF'],
                       help='Reconstruction technique to use')
    return parser.parse_args()
'''
            # Insertar antes de def main():
            content = content.replace('def main():', parser_function + '\\ndef main():')
        
        # 3. Modificar función main para usar argumentos
        if 'args = parse_arguments()' not in content:
            # Buscar el patrón de inicio de main()
            lines = content.split('\\n')
            new_lines = []
            main_found = False
            
            for i, line in enumerate(lines):
                new_lines.append(line)
                if line.strip() == 'def main():' and not main_found:
                    main_found = True
                    # Agregar parsing de argumentos después de def main():
                    new_lines.append('    args = parse_arguments()')
            
            content = '\\n'.join(new_lines)
        
        # 4. Reemplazar stype hardcodeado
        replacements = [
            ('stype = "GAF"  # Modifica según la técnica que quieras evaluar', 'stype = args.technique  # Técnica desde argumentos'),
            ('stype = "GAF"', 'stype = args.technique'),
            ('stype = "RP"', 'stype = args.technique'),
            ('stype = "MTF"', 'stype = args.technique'),
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        # 5. Actualizar la sección de carga de datos reconstruidos para incluir GADF
        gadf_section = '''    data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")
    data_folder_gaf = os.path.join(dataset_folder, "tseries/GAF/sampling_loto/3-fold/fold-0")
    data_folder_mtf = os.path.join(dataset_folder, "tseries/MTF/sampling_loto/3-fold/fold-0")
    data_folder_gadf = os.path.join(dataset_folder, "tseries/GADF/sampling_loto/3-fold/fold-0")
        
    if stype == "RP":
        rec_path = os.path.join(data_folder_rp, "test", "X_all_rec.npy")
        X_all_rec = np.load(rec_path)
        X_recon = X_all_rec
        print("X_recon shape:", X_all_rec.shape)
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
        raise ValueError(f"stype desconocido: {stype}")'''
        
        # Buscar y reemplazar la sección existente
        if 'data_folder_gadf' not in content:
            # Reemplazar la sección completa
            start_marker = 'data_folder_rp = os.path.join(dataset_folder, "tseries/recurrence_plot/sampling_loto/3-fold/fold-0")'
            end_marker = 'raise ValueError("stype desconocido")'
            
            if start_marker in content:
                lines = content.split('\\n')
                new_lines = []
                skip_section = False
                
                for line in lines:
                    if start_marker in line:
                        # Insertar la nueva sección
                        for gadf_line in gadf_section.split('\\n'):
                            new_lines.append(gadf_line)
                        skip_section = True
                    elif 'raise ValueError(' in line and 'stype' in line:
                        skip_section = False
                        # No agregar esta línea, ya está en la nueva sección
                    elif not skip_section:
                        new_lines.append(line)
                
                content = '\\n'.join(new_lines)
        
        # 6. Para scripts de CNN/LSTM/GRU, actualizar el procesamiento de datos
        if any(classifier in script_path for classifier in ['cnn', 'lstm', 'gru']):
            # Asegurar que GADF use el mismo procesamiento que GAF/MTF
            if 'X_train_processed = X_train' in content:
                processing_section = '''    # Transformación de datos según la técnica.
    if stype == "RP":
        # Para RP: se descarta el último time step (de 129 se toman 128)
        X_train_processed = X_train[:, :-1, :]  # (samples, 128, 3)
        X_val_processed = X_val[:, :-1, :]
        input_shape = (128, 3)
    else:
        # Para GAF, GADF, MTF: se asume la forma (129, 3)
        X_train_processed = X_train
        X_val_processed = X_val
        input_shape = (129, 3)'''
        
                # Reemplazar la sección de procesamiento
                old_processing = '''    # Transformación de datos según la técnica.
    # Para "RP": se descarta el último time step (de 129 se toman 128)
    X_train_processed = X_train
    X_val_processed = X_val
    input_shape = (128, 3)'''
                
                content = content.replace(old_processing, processing_section)
        
        # 7. Actualizar mensajes de salida para incluir la técnica
        content = content.replace(
            'print("Evaluación sobre validation_data.npy:")',
            'print(f"Evaluación sobre validation_data.npy: ORIGINALES")'
        )
        content = content.replace(
            'print("Evaluación sobre datos reconstruidos:")',
            'print(f"Evaluación sobre datos reconstruidos: {stype}")'
        )
        
        # Guardar el archivo modificado
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
        
    except Exception as e:
        print(f"❌ Error modificando {script_path}: {str(e)}")
        return False

def run_single_classifier(classifier_name, technique, log_dir, verbose=True):
    """
    Ejecuta un clasificador específico con una técnica específica.
    """
    script_map = {
        'cnn': 'cnnvalidation.py',
        'lstm': 'lstmvalidation.py', 
        'gru': 'gruvalidation.py',
        'knn': 'knnvalidation.py',
        'randomforest': 'randomforestvalidation.py'
    }
    
    if classifier_name not in script_map:
        if verbose:
            print(f"❌ Clasificador desconocido: {classifier_name}")
        return False
        
    script_name = script_map[classifier_name]
    script_path = os.path.join("scripts", script_name)
    log_file = os.path.join(log_dir, f"{classifier_name}_{technique.lower()}.log")
    
    if verbose:
        print(f"🚀 Ejecutando {classifier_name.upper()} con {technique}...")
        print(f"📝 Log: {log_file}")
    
    try:
        start_time = time.time()
        
        # Crear comando - usar ruta absoluta del script
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        full_script_path = os.path.join(scripts_dir, script_name)
        cmd = [sys.executable, full_script_path, "--technique", technique]
        
        # Ejecutar el comando
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {classifier_name.upper()} con {technique} ===\\n")
            f.write(f"Iniciado: {datetime.now()}\\n")
            f.write(f"Comando: {' '.join(cmd)}\\n")
            f.write("="*50 + "\\n\\n")
            
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(scripts_dir)  # El directorio padre de scripts
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            f.write(f"\\n\\n" + "="*50)
            f.write(f"\\nFinalizando: {datetime.now()}")
            f.write(f"\\nDuración: {duration:.2f} segundos")
            f.write(f"\\nCódigo de salida: {result.returncode}")
            
        if result.returncode == 0:
            if verbose:
                print(f"✅ {classifier_name.upper()} con {technique} completado exitosamente")
            return True
        else:
            if verbose:
                print(f"❌ {classifier_name.upper()} con {technique} falló (código: {result.returncode})")
            return False
            
    except Exception as e:
        if verbose:
            print(f"❌ Error ejecutando {classifier_name.upper()} con {technique}: {str(e)}")
        return False

def prepare_validation_scripts():
    """
    Prepara todos los scripts de validación para soportar argumentos y GADF.
    """
    print("🔧 Preparando scripts de validación...")
    
    scripts_to_modify = [
        'cnnvalidation.py',
        'lstmvalidation.py', 
        'gruvalidation.py',
        'knnvalidation.py',
        'randomforestvalidation.py'
    ]
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    success_count = 0
    
    for script_name in scripts_to_modify:
        script_path = os.path.join(scripts_dir, script_name)
        if modify_script_for_gadf_support(script_path):
            print(f"✅ Modificado: {script_name}")
            success_count += 1
        else:
            print(f"❌ Error modificando: {script_name}")
    
    print(f"📋 Scripts modificados: {success_count}/{len(scripts_to_modify)}")
    return success_count == len(scripts_to_modify)

def main():
    """
    Función principal.
    """
    parser = argparse.ArgumentParser(description='Ejecutar clasificadores con técnicas de reconstrucción')
    parser.add_argument('--classifier', type=str, choices=['cnn', 'lstm', 'gru', 'knn', 'randomforest', 'all'],
                       default='all', help='Clasificador a ejecutar')
    parser.add_argument('--technique', type=str, choices=['GAF', 'GADF', 'RP', 'MTF', 'all'],
                       default='all', help='Técnica a usar')
    parser.add_argument('--prepare-only', action='store_true', help='Solo preparar scripts sin ejecutar')
    
    args = parser.parse_args()
    
    print("🎯 Preparador y Ejecutor de Clasificadores")
    print("="*60)
    
    # Crear directorio de logs
    log_dir = "logs_clasificadores"
    os.makedirs(log_dir, exist_ok=True)
    
    # Preparar scripts
    if not prepare_validation_scripts():
        print("❌ Error preparando scripts. Abortando.")
        return
    
    if args.prepare_only:
        print("✅ Scripts preparados. Saliendo (--prepare-only especificado).")
        return
    
    # Determinar qué ejecutar
    if args.classifier == 'all':
        classifiers = ['cnn', 'lstm', 'gru', 'knn', 'randomforest']
    else:
        classifiers = [args.classifier]
    
    if args.technique == 'all':
        techniques = ['GAF', 'GADF', 'RP', 'MTF']
    else:
        techniques = [args.technique]
    
    # Verificar que GADF existe si se va a usar
    if 'GADF' in techniques:
        data_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
        gadf_test_path = os.path.join(data_folder, "tseries/GADF/sampling_loto/3-fold/fold-0/test/X_all_gadf.npy")
        
        if not os.path.exists(gadf_test_path):
            print("⚠️  Datos GADF no encontrados. Se omitirá GADF de la evaluación.")
            techniques = [t for t in techniques if t != 'GADF']
    
    if not techniques:
        print("❌ No hay técnicas válidas para ejecutar.")
        return
    
    # Ejecutar combinaciones
    total_combinations = len(classifiers) * len(techniques)
    current_combination = 0
    results = {}
    
    print(f"\\n📊 Ejecutando {total_combinations} combinaciones...")
    print("Clasificadores:", classifiers)
    print("Técnicas:", techniques)
    print("="*60)
    
    start_time = time.time()
    
    for classifier in classifiers:
        results[classifier] = {}
        
        for technique in techniques:
            current_combination += 1
            print(f"\\n[{current_combination}/{total_combinations}] {classifier.upper()} + {technique}")
            
            success = run_single_classifier(classifier, technique, log_dir)
            results[classifier][technique] = success
            
            print("-" * 40)
    
    # Resumen final
    end_time = time.time()
    total_duration = end_time - start_time
    
    print("\\n" + "="*60)
    print("📋 RESUMEN DE RESULTADOS")
    print("="*60)
    
    success_count = 0
    total_count = 0
    
    for classifier in classifiers:
        print(f"\\n🔸 {classifier.upper()}:")
        for technique in techniques:
            total_count += 1
            status = "✅" if results[classifier][technique] else "❌"
            print(f"   {technique}: {status}")
            if results[classifier][technique]:
                success_count += 1
    
    print(f"\\n📈 ESTADÍSTICAS:")
    print(f"   Exitosos: {success_count}/{total_count}")
    print(f"   Tasa de éxito: {success_count/total_count*100:.1f}%")
    print(f"   Duración total: {total_duration/60:.1f} minutos")
    
    # Guardar resumen
    summary_file = os.path.join(log_dir, "resumen_ejecucion.json")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_minutes": total_duration / 60,
        "total_combinations": total_count,
        "successful": success_count,
        "success_rate": success_count / total_count * 100 if total_count > 0 else 0,
        "results": results,
        "classifiers": classifiers,
        "techniques": techniques
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 Resumen guardado en: {summary_file}")
    print(f"📁 Logs individuales en: {log_dir}/")

if __name__ == "__main__":
    main()
