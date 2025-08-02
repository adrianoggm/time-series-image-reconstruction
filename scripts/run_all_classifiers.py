#!/usr/bin/env python3
"""
Script maestro para ejecutar todos los clasificadores con todas las técnicas de reconstrucción.
Evalúa CNN, LSTM, GRU, KNN y Random Forest con GAF, GADF, RP y MTF.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

def run_classifier_with_technique(classifier_name, technique, log_dir):
    """
    Ejecuta un clasificador específico con una técnica específica.
    
    Args:
        classifier_name (str): Nombre del clasificador ('cnn', 'lstm', 'gru', 'knn', 'randomforest')
        technique (str): Técnica de reconstrucción ('GAF', 'GADF', 'RP', 'MTF')
        log_dir (str): Directorio donde guardar los logs
    """
    script_map = {
        'cnn': 'cnnvalidation.py',
        'lstm': 'lstmvalidation.py', 
        'gru': 'gruvalidation.py',
        'knn': 'knnvalidation.py',
        'randomforest': 'randomforestvalidation.py'
    }
    
    if classifier_name not in script_map:
        print(f"❌ Clasificador desconocido: {classifier_name}")
        return False
        
    script_name = script_map[classifier_name]
    log_file = os.path.join(log_dir, f"{classifier_name}_{technique.lower()}.log")
    
    print(f"🚀 Ejecutando {classifier_name.upper()} con {technique}...")
    print(f"📝 Log: {log_file}")
    
    # Crear comando con técnica específica
    cmd = [
        sys.executable, "-m", f"scripts.{script_name[:-3]}",  # Remover .py
        "--technique", technique
    ]
    
    try:
        start_time = time.time()
        
        # Ejecutar el comando y redirigir output al log
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
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            f.write(f"\\n\\n" + "="*50)
            f.write(f"\\nFinalizando: {datetime.now()}")
            f.write(f"\\nDuración: {duration:.2f} segundos")
            f.write(f"\\nCódigo de salida: {result.returncode}")
            
        if result.returncode == 0:
            print(f"✅ {classifier_name.upper()} con {technique} completado exitosamente")
            return True
        else:
            print(f"❌ {classifier_name.upper()} con {technique} falló (código: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ Error ejecutando {classifier_name.upper()} con {technique}: {str(e)}")
        return False

def modify_validation_scripts():
    """
    Modifica los scripts de validación para que acepten argumentos de línea de comandos.
    """
    print("🔧 Modificating validation scripts para aceptar argumentos...")
    
    scripts_to_modify = [
        'cnnvalidation.py',
        'lstmvalidation.py', 
        'gruvalidation.py',
        'knnvalidation.py',
        'randomforestvalidation.py'
    ]
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    for script_name in scripts_to_modify:
        script_path = os.path.join(scripts_dir, script_name)
        
        if not os.path.exists(script_path):
            print(f"⚠️  Script no encontrado: {script_name}")
            continue
            
        # Leer el script actual
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Agregar imports necesarios si no están presentes
        if 'import argparse' not in content:
            # Encontrar la línea después de los imports existentes
            lines = content.split('\\n')
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    import_end = i + 1
                    
            lines.insert(import_end, 'import argparse')
            content = '\\n'.join(lines)
            
        # Agregar función para parsear argumentos
        parser_function = '''
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run classifier validation')
    parser.add_argument('--technique', type=str, default='GAF',
                       choices=['GAF', 'GADF', 'RP', 'MTF'],
                       help='Reconstruction technique to use')
    return parser.parse_args()
'''
        
        # Insertar la función antes de def main():
        if 'def parse_arguments():' not in content:
            content = content.replace('def main():', parser_function + '\\ndef main():')
            
        # Modificar la función main para usar argumentos
        if 'args = parse_arguments()' not in content:
            content = content.replace(
                'def main():\\n    # Flag',
                'def main():\\n    args = parse_arguments()\\n    # Flag'
            )
            
        # Reemplazar stype hardcodeado con argumento
        content = content.replace(
            'stype = "GAF"  # Modifica según la técnica que quieras evaluar',
            'stype = args.technique  # Técnica pasada por argumento'
        )
        
        # También buscar otras variantes
        content = content.replace(
            'stype = "GAF"',
            'stype = args.technique'
        )
        
        # Guardar el script modificado
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"✅ Modificado: {script_name}")

def main():
    """
    Función principal que ejecuta todos los clasificadores con todas las técnicas.
    """
    print("🎯 Iniciando evaluación completa de clasificadores")
    print("="*60)
    
    # Crear directorio de logs
    log_dir = "logs_clasificadores"
    os.makedirs(log_dir, exist_ok=True)
    
    # Lista de clasificadores y técnicas
    classifiers = ['cnn', 'lstm', 'gru', 'knn', 'randomforest']
    techniques = ['GAF', 'GADF', 'RP', 'MTF']
    
    # Verificar que tenemos los datos reconstruidos para GADF
    data_folder = "C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/"
    gadf_test_path = os.path.join(data_folder, "tseries/GADF/sampling_loto/3-fold/fold-0/test/X_all_gadf.npy")
    
    if not os.path.exists(gadf_test_path):
        print("⚠️  Datos GADF no encontrados. Se omitirá GADF de la evaluación.")
        techniques.remove('GADF')
    
    # Modificar scripts de validación para aceptar argumentos
    modify_validation_scripts()
    
    # Matriz de resultados
    results = {}
    total_combinations = len(classifiers) * len(techniques)
    current_combination = 0
    
    start_time = time.time()
    
    print(f"\\n📊 Ejecutando {total_combinations} combinaciones...")
    print("Clasificadores:", classifiers)
    print("Técnicas:", techniques)
    print("="*60)
    
    # Ejecutar cada combinación
    for classifier in classifiers:
        results[classifier] = {}
        
        for technique in techniques:
            current_combination += 1
            print(f"\\n[{current_combination}/{total_combinations}] {classifier.upper()} + {technique}")
            
            success = run_classifier_with_technique(classifier, technique, log_dir)
            results[classifier][technique] = success
            
            if success:
                print(f"✅ Completado exitosamente")
            else:
                print(f"❌ Falló")
                
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
    
    # Guardar resumen en JSON
    summary_file = os.path.join(log_dir, "resumen_ejecucion.json")
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "duration_minutes": total_duration / 60,
        "total_combinations": total_count,
        "successful": success_count,
        "success_rate": success_count / total_count * 100,
        "results": results,
        "classifiers": classifiers,
        "techniques": techniques
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\\n💾 Resumen guardado en: {summary_file}")
    print(f"📁 Logs individuales en: {log_dir}/")
    
    if success_count == total_count:
        print("\\n🎉 ¡Todas las evaluaciones completadas exitosamente!")
    else:
        print(f"\\n⚠️  {total_count - success_count} evaluaciones fallaron. Revisa los logs para más detalles.")

if __name__ == "__main__":
    main()
