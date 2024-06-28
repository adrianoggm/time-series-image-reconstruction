import os

def calcula_errores(directorio_originales,directorio_creadas,nactivities):
  for i in nactivities:  # Especifica el directorio
    directorio_ori= f"{directorio_originales}/{i}"
    directorio_created=f"{directorio_creadas}/{i}"
    # Lista todos los archivos en el directorio
    archivos_ori = [nombre for nombre in os.listdir(directorio_ori) if os.path.isfile(os.path.join(directorio_ori, nombre))]
    archivos_created = [nombre for nombre in os.listdir(directorio_created) if os.path.isfile(os.path.join(directorio_created, nombre))]
    print(archivos_ori)