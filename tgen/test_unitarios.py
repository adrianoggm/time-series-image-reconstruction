

import numpy as np

def check_values_in_range(matrix, min_value, max_value):
    """
    Comprueba si todos los valores de la matriz están entre min_value y max_value (inclusivos).
    
    Parámetros:
    matrix (np.ndarray): La matriz a comprobar.
    min_value (float): El valor mínimo permitido.
    max_value (float): El valor máximo permitido.
    
    Devuelve:
    bool: True si todos los valores están en el rango [min_value, max_value], False en caso contrario.
    """
    return np.all((matrix >= min_value) & (matrix <= max_value))
def check_dimensions(matrix, expected_shape):
    """
    Comprueba si la matriz tiene las dimensiones especificadas.
    
    Parámetros:
    matrix (np.ndarray): La matriz a comprobar.
    expected_shape (tuple): La forma esperada (dimensiones) de la matriz, por ejemplo, (3, 129).
    
    Devuelve:
    bool: True si la matriz tiene las dimensiones especificadas, False en caso contrario.
    """
    return matrix.shape == expected_shape
def check_binary_matrix(matrix, tolerance=1e-3):
    """
    Comprueba si todos los valores de una matriz float son binarios (cerca de 0.0 o 1.0) 
    dentro de un margen de tolerancia.

    Parámetros:
    matrix (np.ndarray): La matriz de valores float a comprobar.
    tolerance (float): El margen de tolerancia para considerar un valor como 0.0 o 1.0.
    
    Devuelve:
    bool: True si todos los valores son binarios (dentro de la tolerancia), False en caso contrario.
    """
    # Comprobar si todos los valores están cerca de 0.0 o 1.0
    return np.all((np.abs(matrix - 0.0) <= tolerance) | (np.abs(matrix - 1.0) <= tolerance))
def is_unique(vector):
    """Verifica si todos los elementos en un vector son únicos."""
    return len(vector) == len(set(vector))