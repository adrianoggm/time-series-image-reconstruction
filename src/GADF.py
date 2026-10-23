#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import src.activity_data as act
import src.recurrence_plots as rec
#import activity_data as act
#import recurrence_plots as rec
from PIL import Image
import cv2
import time
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from tqdm.auto import trange, tqdm #progress bars for pyhton files (not jupyter notebook)
import os
def normalize_to_zero_one(series):
    """
    Normaliza una serie temporal al rango [0, 1].
    
    Args:
    series (np.array): Serie temporal a normalizar.
    
    Returns:
    np.array: Serie temporal normalizada al rango [0, 1].
    """
    min_val = np.min(series)
    max_val = np.max(series)
    normalized_series = (series - min_val) / (max_val - min_val)
    normalized_series = np.where(normalized_series >= 1., 1., normalized_series)
    normalized_series = np.where(normalized_series <= 0., 0., normalized_series)
    return normalized_series
def tabulate(x, y, f):
    """Return a table of f(x, y): T[i, j] = f(x_j, y_i). Useful for the Gram-like operations.

    The two Gramian kernels run as NumPy ufuncs on the same meshgrid operands (bit-identical to
    np.vectorize over math.cos/math.sin, checked by scripts/_check_encoder_parity.py); any other f
    falls back to np.vectorize.
    """
    X, Y = np.meshgrid(x, y, sparse=True)
    if f is cos_sum:
        return np.cos(X + Y)
    if f is sin_diff:
        return np.sin(X - Y)
    return np.vectorize(f)(X, Y)

def cos_sum(a, b):
    """To work with tabulate."""
    return(math.cos(a+b))
def sin_diff(a, b):
    """To work with tabulate."""
    return(math.sin(a-b))
def normalize_series(series):
    min_val = np.min(series)
    max_val = np.max(series)
    # Floating point inaccuracy!
    scaled_serie =2 * (series - min_val) / (max_val - min_val) - 1
    scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)
    return scaled_serie
def normalize_series07(series, alpha=0.7):
    min_val = np.min(series)
    max_val = np.max(series)
    # Escalar entre -alpha y +alpha
    scaled_serie = 2 * alpha * (series - min_val) / (max_val - min_val) - alpha
    scaled_serie = np.where(scaled_serie >= alpha, alpha, scaled_serie)
    scaled_serie = np.where(scaled_serie <= -alpha, -alpha, scaled_serie)
    return scaled_serie

def gramian_angular_diff_field(series, method='difference', alpha=0.7):
    # Construcción del GADF. Cada variante usa su propio escalado/ángulo:
    if method == 'summation':
        # GASF: amplitud completa [0,1] -> phi in [0, pi/2].
        normalized_series = normalize_to_zero_one(series)
        phi = np.arccos(normalized_series)
        gaf = tabulate(phi, phi, cos_sum)

    elif method == 'difference':
        # GADF restringido: escalado a [-alpha, alpha] (alpha<1/sqrt(2)) ANTES de
        # calcular phi, de modo que todo |phi_i - phi_j| < pi/2 quede dentro de
        # la rama principal de arcsin y el inverso sea cerrado y exacto.
        # (Antes phi se calculaba sobre la normalización [0,1] y el escalado a
        # 0.7 quedaba sin efecto, dejando la imagen justo en el borde pi/2.)
        scaled_series = normalize_series07(series, alpha)
        phi = np.arccos(scaled_series)
        gaf = tabulate(phi, phi, sin_diff)
    else:
        raise ValueError("Method must be 'summation' or 'difference'")

    return gaf
    

def varGADF(data, dim,TIME_STEPS):
    x = []
    k=0
    if dim == 'x':
        k=0
    elif dim == 'y':
        k=1
    elif dim == 'z': 
        k=2
    
    x=np.array(data[:,k]).reshape(1,-1)
    
    X_gaf = gramian_angular_diff_field(x)
    #print(X_gaf)
    
    return X_gaf

def RGBfromGADFMatrix_of_XYZ(X,Y,Z):
    if X.shape != Y.shape or X.shape != Z.shape or Y.shape != Z.shape:
        print('XYZ should be in same shape!')
        return 0
    # One channel per axis; was a per-pixel Python loop with identical values.
    return np.stack([np.asarray(X, dtype=float), np.asarray(Y, dtype=float), np.asarray(Z, dtype=float)], axis=-1)

def SavevarGADF_XYZ(x, sj, item_idx, action=None, normalized=True, path=None, saveImage=True, TIME_STEPS=128):
    if not all([(x==0).all()]):
     _r = varGADF(x,'x', TIME_STEPS)
     _g = varGADF(x,'y', TIME_STEPS)
     _b = varGADF(x,'z', TIME_STEPS)

     
     #print("Y", _g[1][4])
     #print("Z", _b[1][4])
     #print("Y", _g)
     #print("Z", _b)
     
     # plt.close('all')
     # plt.figure(figsize=(1,1))
     # plt.axis('off')
     # plt.margins(0,0)
     # plt.gca().xaxis.set_major_locator(plt.NullLocator())
     # plt.gca().yaxis.set_major_locator(plt.NullLocator())

     #print("fig size: width=", plt.figure().get_figwidth(), "height=", plt.figure().get_figheight())
    
     if normalized:
          #print(_r)
          newImage = RGBfromGADFMatrix_of_XYZ(normalize_to_zero_one(_r), normalize_to_zero_one(_g),normalize_to_zero_one(_b))
          #newImage = RGBfromRPMatrix_of_XYZ(_r, _g, _b)
          # print(newImage.shape)
          #print(newImage[1][4][0]* 255)
          
          newImage = Image.fromarray((np.round(newImage * 255)).astype(np.uint8))
          # plt.imshow(newImage)
          
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure')
               newImage.save(f"{path}{sj}{action}{item_idx}gadf.png")
          # plt.close('all')
     else:
          newImage = RGBfromGADFMatrix_of_XYZ((_r+1)/2, (_g+1)/2,(_b+1)/2)
          newImage = Image.fromarray((newImage * 255).astype(np.uint8))
          # plt.imshow(newImage)
          if saveImage:
               # plt.savefig(f"{path}{sj}{action}{item_idx}.png",bbox_inches='tight',pad_inches = 0, dpi='figure') #dpi='figure' for preserve the correct pixel size (TIMESTEPS x TIMESTEPS)
               newImage.save(f"{path}{sj}{action}{item_idx}gadf.png")
          # plt.close('all')
    
     return newImage
    
    else:
     return None
    
   
def calcular_errores(valores_verdaderos, valores_aproximados):
    # Convertir las listas a arrays de numpy para facilitar los cálculos
    valores_verdaderos = np.array(valores_verdaderos)
    valores_aproximados = np.array(valores_aproximados)
    
    # Calcular el error absoluto
    errores_absolutos = np.abs(valores_verdaderos - valores_aproximados)
    
    # Calcular el error relativo (evitando la división por cero)
    errores_relativos = np.abs(errores_absolutos / valores_verdaderos)
    
    # Calcular el error absoluto promedio
    error_absoluto_promedio = np.mean(errores_absolutos)
    
    # Calcular el error relativo promedio
    error_relativo_promedio = np.mean(errores_relativos)
    
    return error_absoluto_promedio, error_relativo_promedio
def Reconstruct_GADF(img,dictionary,a):
    _r= img[:,:,0].astype('float')
    _g= img[:,:,1].astype('float')
    _b= img[:,:,2].astype('float')
    
    _r=np.interp(_r,(0,255),(-1,1))
    _g=np.interp(_g,(0,255),(-1,1))
    _b=np.interp(_b,(0,255),(-1,1))
    
    r=GADF_to_TS(_r,0,dictionary[a])
    g=GADF_to_TS(_g,1,dictionary[a])
    b=GADF_to_TS(_b,2,dictionary[a])
    N=[]
    N.append(r)
    N.append(g)
    N.append(b)
    return N
def gadf_anchor(gaf, alpha=0.7):
    """Locate the absolute angular anchor of a (transposed) GADF matrix.

    Returns ``(e, phi_e)``: the extreme row index and the boundary angle assigned
    to it. With ``gaf[r][j]=sin(phi_j-phi_r)`` the row with the largest |mean|
    sits on an amplitude boundary; a positive mean means ``phi_r`` is the
    small-angle boundary ``arccos(alpha)`` (x=+alpha), negative the large one.
    """
    row_means = np.mean(gaf, axis=1)
    e = int(np.argmax(np.abs(row_means)))
    phi_e = np.arccos(alpha) if row_means[e] > 0 else np.arccos(-alpha)
    return e, phi_e


def GADF_to_TS(gaf, i, dictionary, method='difference', n_pivots=5, alpha=0.7):
    """Restricted-branch GADF inverse for one axis.

    Recovers the absolute angular trajectory from the off-diagonal entries and
    the boundary anchor, then maps it back to physical units with the stored
    (max, min) calibration.

    Convention: tabulate()/meshgrid store the transposed GADF,
    ``gaf[r][j] = sin(phi_j - phi_r)`` (not ``sin(phi_r - phi_j)``), so every
    arcsin step below ADDS rather than subtracts. Encoder and decoder must stay
    in this same convention.

    The reconstruction is the exact inverse of ``gramian_angular_diff_field``
    under the restricted ``alpha<1/sqrt(2)`` protocol; up to 8-bit PNG
    quantisation the round-trip error is at the resolution of the stored image.
    """
    row_means = np.mean(gaf, axis=1)

    # --- Absolute anchor: the extreme row fixes the free global angular offset.
    e, phi_e = gadf_anchor(gaf, alpha)

    # --- Pivot rows: those with mean closest to zero (near the angular centre).
    # Averaging several reconstructions reduces the noise of relying on a single
    # row. Every pivot is anchored back to the boundary row e:
    #   gaf[e][p] = sin(phi_p - phi_e)  ->  phi_p = phi_e + arcsin(gaf[e][p])
    #   gaf[p][j] = sin(phi_j - phi_p)  ->  phi_j = phi_p + arcsin(gaf[p][j])
    order = np.argsort(np.abs(row_means))
    n_pivots = max(1, min(n_pivots, gaf.shape[0]))
    pivots = order[:n_pivots]

    phi_estimates = []
    for p in pivots:
        phi_p = phi_e + np.arcsin(np.clip(gaf[e][p], -1.0, 1.0))
        phi_j = phi_p + np.arcsin(np.clip(gaf[p], -1.0, 1.0))
        phi_estimates.append(phi_j)
    phi = np.mean(phi_estimates, axis=0)

    # Normalised amplitude in [-alpha, alpha] (= the encoder's normalize_series07).
    scaled = np.cos(phi)

    # --- Denormalisation: exact inverse of normalize_series07, using the stored
    # per-axis (max, min) calibration rather than a min-max rescale of the
    # reconstruction (which would re-anchor on the recovered extremes).
    MAX = dictionary[0]
    MIN = dictionary[1]
    x = MIN[i] + (scaled + alpha) / (2.0 * alpha) * (MAX[i] - MIN[i])

    return x
