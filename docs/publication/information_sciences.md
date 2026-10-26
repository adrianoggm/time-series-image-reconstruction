# Plan de publicación: Information Sciences

Revista objetivo: **Information Sciences** (Elsevier, ISSN 0020-0255, SCIE, JCR Q1 en
Computer Science, Information Systems). Encaja en sus áreas de diseño de algoritmos,
técnicas matemáticas y estadísticas, ingeniería de datos y procesamiento de imagen.
El artículo se presenta como contribución metodológica: los algoritmos inversos, las
condiciones en que funcionan y la evidencia en siete dominios. No como una comparación
de cuatro transformaciones visuales.

## Contribuciones

1. **Fundamentos matemáticos.** Inversión geométrica del Signed RP (MDS 2-D, orientación
   O(2) en forma cerrada a partir del signo, recuperación por solapamiento, centrado sin
   metadatos). Invertibilidad de GASF, invertibilidad restringida de GADF, pérdida en MTF.
   Separación entre reconstrucción exacta, aproximada y con pérdida.
2. **Comparación algorítmica.** Rejilla decodificador × codificación: Hirata y Direct-2D,
   cada uno sobre el RP sin signo y sobre el Signed RP; RP binario clásico como referencia.
   Ablaciones, estabilidad y complejidad. El decodificador RP anterior (adaptación propia
   de Hirata) no se incluye.
3. **Validación transversal.** WISDM, PAMAP2, ESC-50/ESC-10, PTB-XL, Swarm, STEAD y EMG;
   ventanas de 64, 128 y 256 muestras; fidelidad temporal y espectral; utilidad en
   clasificación con seis clasificadores; coste.

## Requisitos de la revista que afectan al trabajo

| Requisito | Qué hacemos |
|---|---|
| Artículo original de investigación | Contribución metodológica |
| ~40 páginas a doble espacio (orientativo) | Lo principal en el artículo; el resto en el suplemento |
| ~8 figuras y tablas (orientativo) | 5 figuras compuestas y 3 tablas |
| Abstract | 200 palabras como máximo |
| Keywords | 6–7 |
| Highlights | 3–5, de 85 caracteres como máximo, en fichero aparte |
| Formato | LaTeX con la plantilla oficial de Elsevier (`elsarticle`) |
| Revisión | Simple anónima: los autores se identifican |
| Datos (opción C) | Depositar y enlazar código y datos derivados, o justificar por qué no |
| Autoría | CRediT, afiliaciones, autor de correspondencia, biografía y foto |
| IA generativa | Declarar su uso; ecuaciones, referencias y resultados verificados por los autores |

## Figuras y tablas previstas

| Elemento | Contenido |
|---|---|
| Figura 1 | Esquema codificación → imagen → reconstrucción |
| Figura 2 | Ejemplos de RP, GASF, GADF, MTF y XIRP |
| Figura 3 | Signed RP frente a RP sin signo y frente a Hirata |
| Figura 4 | Reconstrucciones originales y recuperadas en varios dominios |
| Figura 5 | Robustez y coste frente a la longitud de ventana |
| Tabla 1 | Características de los siete datasets |
| Tabla 2 | Fidelidad de los decodificadores en todos los datasets |
| Tabla 3 | Ablación de RP, utilidad en clasificación y síntesis estadística |

## Afirmación a acotar

«RP, GASF y GADF son exactos en coma flotante y todo el error procede de la cuantización
de 8 bits» solo vale con hipótesis explícitas: calibración (max, min) por canal y ventanas
no degeneradas. Casos documentados en los que no se cumple: ventanas casi planas (una de
ESC-50), estados casi colineales (Swarm a 1 Hz) y amplitudes diminutas (canales muertos de
STEAD, resuelto con la normalización de escala).

## Milestones

- **M9 · Contribución y afirmaciones** (#120–#125): novedad frente a Hirata, condiciones de
  invertibilidad, alcance de la afirmación de los 8 bits, complejidad, orientación sin la
  señal original, referencias.
- **M10 · Evidencia experimental final** (#126–#129, más #87 y #90 de M7): ESC-10, fracción
  de pares informativos para el signo, robustez multi-dataset, síntesis estadística.
- **M11 · Manuscrito** (#130–#136): plantilla, título y abstract, figuras, tablas,
  suplemento, highlights, revisión interna.
- **M12 · Envío** (#137–#141): depósito de código y datos, declaraciones, autores, carta al
  editor, preprint y modalidad.
- **M8 · Reproducibilidad** (ya abierto): manifiestos, hashes, CI, script de reproducción.

Orden: cerrar M9 y M10 antes de redactar M11. Los siete datasets y la comparación de RP
pueden cambiar tablas, conclusiones y abstract.
