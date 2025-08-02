# Resultados de Análisis de Errores de Reconstrucción

## Resumen Ejecutivo

Este documento presenta los resultados del análisis de errores para cuatro métodos de codificación y reconstrucción de series temporales:
- **GAF (GASF)**: Gramian Angular Summation Field
- **GADF**: Gramian Angular Difference Field  
- **RP**: Recurrence Plots
- **MTF**: Markov Transition Field

Los métodos fueron evaluados usando 5,484 muestras del dataset WISDM con series temporales de 128 puntos en 3 dimensiones (x, y, z).

## Tabla Comparativa de Errores Globales

| Método | Error Absoluto Promedio | Error Relativo Promedio | Error Cuadrático Medio | Coeficiente Correlación Pearson |
|--------|------------------------|------------------------|------------------------|--------------------------------|
| **GAF** | **0.0210** ± 0.0238 | **0.0239** ± 0.1114 | **0.0045** ± 0.0131 | **0.9999** ± 0.0002 |
| **GADF** | 1.4684 ± 1.6508 | 2.2236 ± 8.1629 | 5.5783 ± 13.1844 | 0.9832 ± 0.0283 |
| **RP** | 3.8718 ± 4.0130 | 4.2814 ± 16.7643 | 41.4849 ± 79.0055 | 0.2945 ± 0.4245 |
| **MTF** | 5.1750 ± 5.8583 | 6.3485 ± 20.9817 | 70.8220 ± 127.1354 | 0.0117 ± 0.6205 |

### Ranking de Performance (Mejor a Peor)
1. 🥇 **GAF** - Excelente en todos los métricas
2. 🥈 **GADF** - Buena correlación, errores moderados  
3. 🥉 **RP** - Correlación baja, errores altos
4. 🔴 **MTF** - Correlación muy pobre, errores muy altos

---

## Análisis Detallado por Dimensión

### Error Absoluto Promedio por Dimensión

| Método | Dimensión X | Dimensión Y | Dimensión Z |
|--------|-------------|-------------|-------------|
| **GAF** | 0.0245 ± 0.0260 | 0.0198 ± 0.0227 | 0.0186 ± 0.0223 |
| **GADF** | 1.4759 ± 1.4730 | 1.4409 ± 1.6116 | 1.4884 ± 1.8460 |
| **RP** | 4.1799 ± 4.0057 | 4.0628 ± 4.2368 | 3.3727 ± 3.7331 |
| **MTF** | 5.3557 ± 5.6519 | 5.5184 ± 6.2796 | 4.6508 ± 5.5820 |

### Error Relativo Promedio por Dimensión

| Método | Dimensión X | Dimensión Y | Dimensión Z |
|--------|-------------|-------------|-------------|
| **GAF** | 0.0170 ± 0.0715 | 0.0144 ± 0.0514 | 0.0404 ± 0.1708 |
| **GADF** | 1.0934 ± 6.1122 | 2.1246 ± 9.4770 | 3.4598 ± 8.3622 |
| **RP** | 2.7439 ± 18.3713 | 3.6365 ± 14.8749 | 6.4765 ± 16.6373 |
| **MTF** | 3.5317 ± 15.8679 | 6.0450 ± 20.3115 | 9.4861 ± 25.2911 |

### Error Cuadrático Medio por Dimensión

| Método | Dimensión X | Dimensión Y | Dimensión Z |
|--------|-------------|-------------|-------------|
| **GAF** | 0.0054 ± 0.0125 | 0.0050 ± 0.0132 | 0.0032 ± 0.0134 |
| **GADF** | 4.8420 ± 9.8417 | 5.8641 ± 11.9266 | 6.0287 ± 16.7795 |
| **RP** | 46.2094 ± 77.1970 | 46.5580 ± 89.5127 | 31.6713 ± 67.8845 |
| **MTF** | 78.2240 ± 134.5760 | 78.6968 ± 139.9129 | 55.5372 ± 102.2523 |

### Coeficiente de Correlación de Pearson por Dimensión

| Método | Dimensión X | Dimensión Y | Dimensión Z |
|--------|-------------|-------------|-------------|
| **GAF** | **0.9999** ± 0.0002 | **0.9999** ± 0.0002 | **0.9999** ± 0.0002 |
| **GADF** | 0.9833 ± 0.0280 | 0.9847 ± 0.0321 | 0.9816 ± 0.0243 |
| **RP** | 0.3062 ± 0.4367 | 0.2834 ± 0.4200 | 0.2937 ± 0.4163 |
| **MTF** | 0.0341 ± 0.6201 | 0.0006 ± 0.6240 | 0.0005 ± 0.6167 |

---

## Estadísticas Extremas

### Valores Mínimos y Máximos del Error Cuadrático

| Método | Mínimo | Máximo | Rango |
|--------|--------|--------|-------|
| **GAF** | 5.16e-09 | 0.2393 | 0.2393 |
| **GADF** | 2.76e-05 | 150.58 | 150.58 |
| **RP** | 1.55e-04 | 1637.70 | 1637.70 |
| **MTF** | 3.63e-04 | 3495.49 | 3495.49 |

### Valores Mínimos y Máximos de Correlación de Pearson

| Método | Mínimo | Máximo | Rango |
|--------|--------|--------|-------|
| **GAF** | 0.9965 | 0.9999 | 0.0034 |
| **GADF** | 0.4313 | 0.9999 | 0.5686 |
| **RP** | -0.6327 | 0.9620 | 1.5947 |
| **MTF** | -0.9699 | 0.9432 | 1.9131 |

---

## Interpretación de Resultados

### 🏆 GAF (Ganador Absoluto)
- **Excelencia**: Correlación casi perfecta (0.9999)
- **Precisión**: Errores extremadamente bajos
- **Consistencia**: Baja variabilidad en todas las métricas
- **Recomendación**: Método óptimo para reconstrucción

### 🥈 GADF (Segundo Lugar)
- **Buena Performance**: Correlación alta (0.9832)
- **Errores Moderados**: Aceptables para muchas aplicaciones
- **Estabilidad**: Menos consistente que GAF pero superior a RP y MTF

### 🥉 RP (Tercer Lugar)
- **Performance Limitada**: Correlación baja (0.2945)
- **Errores Significativos**: Altos valores de error en todas las métricas
- **Uso Específico**: Puede ser útil para análisis cualitativos

### 🔴 MTF (Último Lugar)
- **Performance Pobre**: Correlación muy baja (0.0117)
- **Errores Muy Altos**: Los peores valores en todas las métricas
- **No Recomendado**: Para aplicaciones que requieren alta fidelidad

---

## Conclusiones

1. **GAF es claramente superior** para tareas de reconstrucción de series temporales
2. **GADF ofrece un compromiso razonable** entre calidad y complejidad
3. **RP y MTF no son recomendables** para reconstrucción de alta fidelidad
4. **La dimensión Z** tiende a tener errores ligeramente mayores en todos los métodos
5. **La variabilidad** es significativamente menor en GAF comparado con otros métodos

## Recomendaciones

- **Para aplicaciones críticas**: Usar GAF exclusivamente
- **Para análisis exploratorio**: GADF puede ser suficiente
- **Para visualización**: RP puede ser útil pese a sus limitaciones
- **Evitar MTF**: Para reconstrucción cuantitativa de series temporales

---

*Análisis realizado con 5,484 muestras del dataset WISDM (sampling LOTO, 3-fold, fold-0)*
