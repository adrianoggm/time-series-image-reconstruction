import os
import numpy as np
import sklearn.metrics as metrics
def generate_all_error(directorio_originales,directorio_creadas,nactivities):
  for i in nactivities:  # Especifica el directorio
    directorio_ori= f"{directorio_originales}/{i}"
    directorio_created=f"{directorio_creadas}/{i}"
    # Lista todos los archivos en el directorio
    archivos_ori = [nombre for nombre in os.listdir(directorio_ori) if os.path.isfile(os.path.join(directorio_ori, nombre))]
    archivos_created = [nombre for nombre in os.listdir(directorio_created) if os.path.isfile(os.path.join(directorio_created, nombre))]
    print(archivos_ori)

def calcular_errores(valores_verdaderos, valores_aproximados):
    # Convertir las listas a arrays de numpy para facilitar los cálculos
    valores_verdaderos = np.array(valores_verdaderos)
    valores_aproximados = np.array(valores_aproximados)
    
    # Calcular el error absoluto
    error_absoluto_promedio = metrics.mean_absolute_error(valores_verdaderos,valores_aproximados)
  
    
    # Calcular el error relativo promedio
    error_relativo_promedio = metrics.mean_absolute_percentage_error(valores_verdaderos,valores_aproximados)
  
    
    return error_absoluto_promedio, error_relativo_promedio

def rmse(valores_verdaderos, valores_aproximados):
    error= metrics.mean_squared_error(valores_verdaderos,valores_aproximados)
    return error
def stderror(valores_verdaderos, valores_aproximados):
    # Calcular el error absoluto
    error = valores_verdaderos - valores_aproximados
    error=np.std(error,ddof=1)
    return error

def pointwise_global(orig, rec):
    """Global pointwise metrics (MAE, MRE, MSE, Pearson) between originals and
    reconstructions, both shaped ``(N, T, C)``.

    Single source of truth for the paper's error-table numbers: both
    ``scripts/_verify_error_tables.py`` and ``scripts/search_rp_rho.py`` call
    this so the reported errors and the rho search measure identically.
    """
    diff = orig - rec
    mae = float(np.abs(diff).mean())
    mse = float((diff ** 2).mean())
    safe = np.where(np.abs(orig) < 1.0, 1.0, np.abs(orig))
    mre = float((np.abs(diff) / safe).mean())
    oc = orig - orig.mean(axis=1, keepdims=True)
    rc = rec - rec.mean(axis=1, keepdims=True)
    num = (oc * rc).sum(axis=1)
    den = np.linalg.norm(oc, axis=1) * np.linalg.norm(rc, axis=1)
    pe = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0).mean()
    return mae, mre, mse, float(pe)


##Calculates all different error measures for each channel returning them in n-channel tuples
def ts_error(original,creada,flag=True):
    errores_absolutos=[]
    errores_relativos=[]
    errores_rms=[]
    errores_std=[]
    errores_pearson =[]
    t=0
    if len(original)!=len(creada[1]):
      #print("entra",len(original),len(creada))
      for i in range(0,3):
        pearsona=np.corrcoef(original[1:,i], creada[i])[0,1]
        pearsonb=np.corrcoef(original[:-1,i], creada[i])[0,1]
        if pearsonb<=pearsona :
          error_absoluto, error_relativo = calcular_errores(original[1:,i], creada[i])
          #d = dtw.distance_fast(original[:-1,i], creada[i], use_pruning=True)
          rms=rmse(original[1:,i], creada[i])
          std=stderror(original[1:,i], creada[i])
          pearson=pearsona

        else :
          error_absoluto, error_relativo = calcular_errores(original[:-1,i], creada[i])
          #d = dtw.distance_fast(original[:-1,i], creada[i], use_pruning=True)
          rms=rmse(original[:-1,i], creada[i])
          std=stderror(original[:-1,i], creada[i])
          pearson=pearsonb
          t+=1
        #print(f"Error Absoluto Promedio: {error_absoluto}")
        #print(f"Error Relativo Promedio: {error_relativo}")
        #print(f"Error DTW: {d}")
        #print(f"Coeficiente de correlación: {pearson}")
        errores_absolutos=np.append(errores_absolutos,error_absoluto)
        errores_relativos=np.append(errores_relativos,error_relativo)
        errores_rms=np.append(errores_rms,rms)
        errores_std=np.append(errores_std,std)
        errores_pearson=np.append(errores_pearson,pearson)
    else:
       if flag:
          for i in range(0,3):
            pearson=np.corrcoef(original[:,i], creada[i])[0,1]
            error_absoluto, error_relativo = calcular_errores(original[:,i], creada[i])
            #d = dtw.distance_fast(original[:-1,i], creada[i], use_pruning=True)
            rms=rmse(original[:,i], creada[i])
            std=stderror(original[:,i], creada[i])
            errores_absolutos=np.append(errores_absolutos,error_absoluto)
            errores_relativos=np.append(errores_relativos,error_relativo)
            errores_rms=np.append(errores_rms,rms)
            errores_std=np.append(errores_std,std)
            errores_pearson=np.append(errores_pearson,pearson)
       else :
            for i in range(0,3):
              minv=creada[i]*-1
              pearsona=np.corrcoef(original[:,i], minv)[0,1]
              pearsonb=np.corrcoef(original[:,i], creada[i])[0,1]
              if pearsonb<=pearsona :
                error_absoluto, error_relativo = calcular_errores(original[:,i], minv)
                #d = dtw.distance_fast(original[:-1,i], creada[i], use_pruning=True)
                rms=rmse(original[:,i], minv)
                std=stderror(original[:,i], minv)
                pearson=np.abs(pearsona)
                t+=1
              else :
                error_absoluto, error_relativo = calcular_errores(original[:,i], creada[i])
                #d = dtw.distance_fast(original[:-1,i], creada[i], use_pruning=True)
                rms=rmse(original[:,i], creada[i])
                std=stderror(original[:,i], creada[i])
                pearson=np.abs(pearsonb)
                
              #print(f"Error Absoluto Promedio: {error_absoluto}")
              #print(f"Error Relativo Promedio: {error_relativo}")
              #print(f"Error DTW: {d}")
              #print(f"Coeficiente de correlación: {pearson}")
              errores_absolutos=np.append(errores_absolutos,error_absoluto)
              errores_relativos=np.append(errores_relativos,error_relativo)
              errores_rms=np.append(errores_rms,rms)
              errores_std=np.append(errores_std,std)
              errores_pearson=np.append(errores_pearson,pearson)
        
    return  errores_absolutos, errores_relativos,errores_rms,errores_std,errores_pearson,t  
