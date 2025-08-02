#!/usr/bin/env python3
"""
Script para monitorear el progreso de la ejecución de clasificadores.
"""

import os
import json
import time
from datetime import datetime

def check_progress():
    """
    Revisa el progreso de la ejecución consultando los logs y el archivo de resumen.
    """
    print("🔍 Monitor de Progreso - Ejecución de Clasificadores")
    print("="*60)
    
    # Verificar archivos de log
    log_dir = "logs_clasificadores"
    if not os.path.exists(log_dir):
        print("❌ Directorio de logs no encontrado. ¿Se inició la ejecución?")
        return
    
    # Contar archivos de log existentes
    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
    print(f"📁 Archivos de log encontrados: {len(log_files)}")
    
    # Mostrar archivos de log recientes
    if log_files:
        print("\\n📋 Últimos logs:")
        for log_file in sorted(log_files)[-5:]:  # Últimos 5
            log_path = os.path.join(log_dir, log_file)
            stat = os.stat(log_path)
            size_kb = stat.st_size / 1024
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S")
            print(f"   {log_file:<25} | {size_kb:>6.1f} KB | {mod_time}")
    
    # Verificar archivo de resumen
    summary_file = os.path.join(log_dir, "resumen_ejecucion.json")
    if os.path.exists(summary_file):
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            print("\\n📊 PROGRESO ACTUAL:")
            print(f"   ⏱️  Duración: {summary.get('duration_minutes', 0):.1f} minutos")
            print(f"   ✅ Exitosos: {summary.get('successful', 0)}")
            print(f"   📊 Total: {summary.get('total_combinations', 0)}")
            print(f"   🎯 Tasa éxito: {summary.get('success_rate', 0):.1f}%")
            
            # Mostrar detalles de resultados
            results = summary.get('results', {})
            if results:
                print("\\n🔸 DETALLES POR CLASIFICADOR:")
                for classifier, techniques in results.items():
                    successful_techniques = [t for t, success in techniques.items() if success]
                    total_techniques = len(techniques)
                    print(f"   {classifier.upper():<12}: {len(successful_techniques)}/{total_techniques} técnicas completadas")
                    if successful_techniques:
                        print(f"   {'':>14} ✅ {', '.join(successful_techniques)}")
                    failed_techniques = [t for t, success in techniques.items() if not success]
                    if failed_techniques:
                        print(f"   {'':>14} ❌ {', '.join(failed_techniques)}")
        
        except Exception as e:
            print(f"❌ Error leyendo resumen: {str(e)}")
    else:
        print("⚠️  Archivo de resumen no encontrado aún.")
    
    # Verificar el log principal
    main_log = "logs_all_classifiers.log"
    if os.path.exists(main_log):
        stat = os.stat(main_log)
        size_kb = stat.st_size / 1024
        print(f"\\n📄 Log principal: {main_log} ({size_kb:.1f} KB)")
        
        # Mostrar últimas líneas
        try:
            with open(main_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    print("\\n📝 Últimas líneas del log principal:")
                    for line in lines[-5:]:
                        print(f"   {line.strip()}")
        except Exception as e:
            print(f"❌ Error leyendo log principal: {str(e)}")
    
    print(f"\\n🔄 Última actualización: {datetime.now().strftime('%H:%M:%S')}")

def monitor_continuously():
    """
    Monitorea continuamente cada 30 segundos.
    """
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')  # Limpiar pantalla
            check_progress()
            print("\\n💤 Esperando 30 segundos... (Ctrl+C para salir)")
            time.sleep(30)
    except KeyboardInterrupt:
        print("\\n\\n👋 Monitor detenido por el usuario.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
        monitor_continuously()
    else:
        check_progress()
