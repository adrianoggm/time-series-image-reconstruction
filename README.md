# Official Implementation: TFG Diseño e implementación de la reconstrucción de series temporales a partir de imágenes

[![GitHub stars](https://img.shields.io/github/stars/frangam/diff-tsd.svg)](https://github.com/frangam/diff-tsd/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/frangam/diff-tsd.svg)](https://github.com/frangam/diff-tsd/network/members)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT) <img src="https://img.shields.io/github/release/frangam/diff-tsd.svg"/> [![GitHub all releases](https://img.shields.io/github/downloads/frangam/diff-tsd/total)](https://github.com/frangam/diff-tsd/releases/download/1.0/diff-tsd.zip)

[[`Paper`]Coming Soon] [[`Dataset`](#our-recurrence-plots-dataset)] [[`Citation`](#citation)]



Welcome to the official implementation repository of the TFG "Diseño e implementación de la reconstrucción de series temporales a partir de imágenes" based on paper titled "Diff-TSD: Modelling Time-series Data Generation with Diffusion Models". This repository provides detailed insights, datasets, and other essential resources related to our research and findings.

## Introduction

In the TFG "Diseño e implementación de la reconstrucción de series temporales a partir de imágenes", we explore the potential of reconstruction of the different image codification techniques and intricacies of generating time-series data using diffusion models. As an integral part of this work, this repository serves as a comprehensive platform to access the datasets, recurrence plots,grammian angular fields, markov transition fields and other relevant resources that were instrumental in our research.


## Dataset
We used WISDM dataset. The [WISDM dataset](https://ieeexplore.ieee.org/document/8835065/) focuses on data from smartwatch wearables. Our study particularly delves into the dataset which records 51 subjects performing 18 daily activities, such as "walking" and "jogging". We've honed in on five non-hand-oriented activities: "walking", "jogging", "stairs" (both ascending and descending), "sitting", and "standing", amassing a total of 1,053,141 instances. Data collection utilized the smartwatch's accelerometer, recording at a frequency of 20 Hz. For a visual representation of the acceleration waveforms for each activity.

### Activity Distribution of WISDM

| Activity | Instances | Percentage |
|----------|-----------|------------|
| Standing | 216,529   | 20.6%      |
| Sitting  | 213,018   | 20.2%      |
| Walking  | 210,495   | 20.0%      |
| Stairs   | 207,312   | 19.7%      |
| Jogging  | 205,787   | 19.5%      |

As is common in various studies [^1^] utilizing this dataset for classification tasks, we opted to segment the data into full non-overlapping windowed segments (FNOW). Each segment contains 129 data points. The choice of 129, while seemingly unusual, is intentional; one additional data point beyond the typical 128 [^1^] allows us to create recurrence plots of 128x128 pixels.

[^1^]: Abdel-Salam, H., Pham, T., & Elgendi, M. (2021). A Benchmark Dataset and Evaluation of Human Activity Recognition on Smartphones. *Journal of Information*.



## Execution in background 
You can run the scripts with [nohup](https://en.wikipedia.org/wiki/Nohup) which ignores the hangup signal. This means that you can close the terminal without stopping the execution. Also, don’t forget to add & so the script runs in the background:

```sh
$ nohup accelerate launch train.py --config CONFIG_FILE  > your.log &
```

In addition, to close remote terminal safely, you have to run the exit commant, and do not close manually the terminal:

```sh
$ exit
```

Finally, you can identify the running processes of training script with:
```sh
$ ps ax | grep train.py
```

or list of all running processes of Python:
```sh
$ ps -fA | grep python
```

then, kill the desired one:

```sh
$ kill PID
```

## 1. Create Data Splits
In all bash command, we can combine the use "nohup" command to execute a script withouth interuptions (avoiding terminal disconnections, etc.) and "&" symbol at the end of the command for a background execution. We also can use "> filename.log" to put the results in a log file.

**Sampling techniques**
- The Leave-One-Trial-Out (LOTO) approach is a cutting-edge method in sample generation. Each trial encompasses a unique raw activity signal for a single subject, ensuring an impartial evaluation and facilitating the creation of a sufficient number of samples. Additionally, this technique prevents the duplication of trials with identical raw signals (trials of the same label) across both training and testing datasets.
- The Leave-One-Subject-Out (LOSO) approach is a sampling technique inspired by the Leave-One-Trial-Out method. In this approach, all trials belonging to a single subject are considered as an indivisible unit, ensuring that there are no trials from the same subject duplicated in the training and testing datasets. This technique maintains data integrity and prevents potential biases caused by the presence of trials from the same subject in both datasets, allowing for a more robust and reliable evaluation of the model's performance. This technique is the most strict, which proposes a subject-wise approach instead record-wise, and in the literature is not commonly assessed, maybe due to its resulting lower accuracy.


### Our Recurrence plots Dataset

We performed two experiments: One using LOTO to compare our results with previous results and the other using LOSO.

The table below presents the 3-fold data distribution for each sampling approach:

The table below presents the 3-fold data distribution for each sampling approach:

<table>
  <thead>
    <tr>
      <th rowspan="2">3-Folds</th>
      <th colspan="3">FNOW + LOTO</th>
      <th colspan="3">FNOW + LOSO</th>
    </tr>
    <tr>
      <th>Train samples</th>
      <th>Test samples</th>
      <th>Total</th>
      <th>Train samples</th>
      <th>Test samples</th>
      <th>Total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Fold-1</td>
      <td>5392</td>
      <td>2672</td>
      <td>8064</td>
      <td>5408</td>
      <td>2688</td>
      <td>8096</td>
    </tr>
    <tr>
      <td>Fold-2</td>
      <td>5392</td>
      <td>2688</td>
      <td>8080</td>
      <td>5344</td>
      <td>2768</td>
      <td>8112</td>
    </tr>
    <tr>
      <td>Fold-3</td>
      <td>5392</td>
      <td>2672</td>
      <td>8064</td>
      <td>5456</td>
      <td>2640</td>
      <td>8096</td>
    </tr>
  </tbody>
</table>


So, from the WISDM dataset, we extracted [Recurrence plots](https://ieeexplore.ieee.org/document/8691521/) with a legnth of 129 points (128x128 pixels) were generated for each of the five selected classes across every fold. These plots, inspired by the work of Lu and Tong in "Robust Single Accelerometer-Based Activity Recognition Using Modified Recurrence Plot", are available for download on the Hugging Face platform.



Here, an example of the reccurente plots dataset:

<table>
<tr>
    <td>
        <img src="doc/model_fold_0_class_0_demo_00669000.png" alt="Class 0 Image" width="250"/><br>
        <center>Class 0 (Walking)</center>
    </td>
    <td>
        <img src="doc/model_fold_0_class_1_demo_00659000.png" alt="Class 1 Image" width="250"/><br>
        <center>Class 1 (Jogging)</center>
    </td>
    <td>
        <img src="doc/model_fold_0_class_2_demo_00669000.png" alt="Class 2 Image" width="250"/><br>
        <center>Class 2 (Stairs)</center>
    </td>
</tr>
<tr>
    <td>
        <img src="doc/model_fold_0_class_3_demo_00679000.png" alt="Class 3 Image" width="250"/><br>
        <center>Class 3 (Sitting)</center>
    </td>
    <td>
        <img src="doc/model_fold_0_class_4_demo_00689000.png" alt="Class 4 Image" width="250"/><br>
        <center>Class 4 (Standing)</center>
    </td>
</tr>
</table>



### Create Images (RP,GASF,MTF)
If you want to create recurrence plots:

- "--create-numpies" is for create the first time the numpy arrays.
- With "--sampling loto" you can select the sampling method "loto" or "loso" with "--sampling loso".
- With "--image type " you can select the image construction or reconstruction method "--sampling loso".
For LOTO approach:
```sh
 $ nohup ./generate_images.py  --data-name WISDM --n-folds 3 --image-type GAF --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loto > generate_images.log &

```


### Reconstruct Images (RP,GASF,MTF)
Then, we have to execute once generated the images  the reconstruction of the time series:

```sh
 $ nohup  ./generate_time_series.py --data-name WISDM --n-folds 3 --image-type GAF --data-folder /home/adriano/Escritorio/TFG/data/WISDM/  --sampling loto > ts_plots_loto.log &

```




---


Generates all diferent RP GAF MTF IMAGES for LOTO  aproach
python -m scripts.generate_images --data-name WISDM --n-folds 3 --image-type MTF --data-folder C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/ --sampling loto > generate_images.log


 python -m scripts.generate_time_series --data-name WISDM --n-folds 3 --image-type GAF --data-folder C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/ --sampling loto > ts_plots_loto_mtf.log 

 python -m scripts.generate_time_series --data-name WISDM --n-folds 3 --image-type GAF --data-folder C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/ --sampling loto > ts_plots_loto_gaf.log 

 python -m scripts.generate_time_series --data-name WISDM --n-folds 3 --image-type RP --data-folder C:/Users/adria/OneDrive/Documentos/TFG-Tratamiento-de-Series-Temporales/data/WISDM/ --sampling loto > ts_plots_loto_rp.log 

 Evaluación sobre validation_data.npy: ORIGINALES
              precision    recall  f1-score   support

         0.0       0.80      0.95      0.87       543
         1.0       0.99      0.98      0.98       532
         2.0       0.90      0.75      0.82       536
         3.0       0.89      0.93      0.91       551
         4.0       0.94      0.88      0.91       559

    accuracy                           0.90      2721
   macro avg       0.90      0.90      0.90      2721
weighted avg       0.90      0.90      0.90      2721

86/86 [==============================] - 1s 7ms/step
Evaluación sobre datos reconstruidos: RP
              precision    recall  f1-score   support

         0.0       0.68      0.34      0.45       543
         1.0       0.73      0.89      0.80       532
         2.0       0.48      0.70      0.57       536
         3.0       0.83      0.79      0.81       551
         4.0       0.87      0.76      0.81       559

    accuracy                           0.70      2721
   macro avg       0.72      0.70      0.69      2721
weighted avg       0.72      0.70      0.69      2721
86/86 [==============================] - 1s 6ms/step
Evaluación sobre datos reconstruidos: MTF
              precision    recall  f1-score   support

         0.0       0.60      0.47      0.52       543
         1.0       0.58      0.97      0.73       532
         2.0       0.48      0.43      0.45       536
         3.0       0.92      0.81      0.86       551
         4.0       0.96      0.76      0.85       559

    accuracy                           0.69      2721
   macro avg       0.71      0.69      0.68      2721
weighted avg       0.71      0.69      0.68      2721

86/86 [==============================] - 1s 7ms/step
Evaluación sobre datos reconstruidos: GAF
              precision    recall  f1-score   support

         0.0       0.80      0.95      0.87       543
         1.0       0.99      0.98      0.98       532
         2.0       0.90      0.76      0.82       536
         3.0       0.89      0.93      0.91       551
         4.0       0.94      0.88      0.91       559

    accuracy                           0.90      2721
   macro avg       0.90      0.90      0.90      2721
weighted avg       0.90      0.90      0.90      2721

---
KNN
Evaluación sobre validation_data.npy: ORGINAL
              precision    recall  f1-score   support

         0.0       0.75      0.49      0.59       543
         1.0       1.00      0.77      0.87       532
         2.0       0.54      0.05      0.09       536
         3.0       0.79      0.95      0.86       551
         4.0       0.41      0.92      0.57       559

    accuracy                           0.64      2721
   macro avg       0.70      0.64      0.60      2721
weighted avg       0.70      0.64      0.60      2721

(2721, 129, 3)
(2721, 387)
Evaluación sobre datos reconstruidos: RP
              precision    recall  f1-score   support

         0.0       0.73      0.25      0.37       543
         1.0       0.92      0.52      0.67       532
         2.0       0.35      0.03      0.05       536
         3.0       0.57      0.88      0.69       551
         4.0       0.36      0.85      0.50       559

    accuracy                           0.51      2721
   macro avg       0.58      0.51      0.46      2721
weighted avg       0.58      0.51      0.46      2721

Evaluación sobre datos reconstruidos:MTF
              precision    recall  f1-score   support

         0.0       0.59      0.47      0.52       543
         1.0       0.99      0.34      0.51       532
         2.0       0.41      0.10      0.16       536
         3.0       0.62      0.92      0.74       551
         4.0       0.43      0.88      0.58       559

    accuracy                           0.55      2721
   macro avg       0.61      0.54      0.50      2721
weighted avg       0.61      0.55      0.50      2721

Evaluación sobre datos reconstruidos: GAF
              precision    recall  f1-score   support

         0.0       0.75      0.49      0.59       543
         1.0       1.00      0.77      0.87       532
         2.0       0.53      0.05      0.09       536
         3.0       0.79      0.95      0.86       551
         4.0       0.41      0.92      0.57       559

    accuracy                           0.64      2721
   macro avg       0.70      0.64      0.60      2721
weighted avg       0.69      0.64      0.60      2721

LSTM

Evaluación sobre validation_data.npy: ORIGINAL
              precision    recall  f1-score   support

         0.0       0.77      0.86      0.82       543
         1.0       0.99      0.99      0.99       532
         2.0       0.84      0.71      0.77       536
         3.0       0.87      0.85      0.86       551
         4.0       0.83      0.87      0.85       559

    accuracy                           0.86      2721
   macro avg       0.86      0.86      0.86      2721
weighted avg       0.86      0.86      0.86      2721

86/86 [==============================] - 1s 9ms/step
Evaluación sobre datos reconstruidos: RP
              precision    recall  f1-score   support

         0.0       0.59      0.22      0.32       543
         1.0       0.68      0.88      0.76       532
         2.0       0.43      0.60      0.50       536
         3.0       0.79      0.77      0.78       551
         4.0       0.79      0.75      0.77       559

    accuracy                           0.65      2721
   macro avg       0.65      0.65      0.63      2721
weighted avg       0.66      0.65      0.63      2721

86/86 [==============================] - 1s 9ms/step MTF
Evaluación sobre datos reconstruidos:
              precision    recall  f1-score   support

         0.0       0.63      0.50      0.56       543
         1.0       0.59      1.00      0.74       532
         2.0       0.56      0.43      0.49       536
         3.0       0.91      0.80      0.85       551
         4.0       0.89      0.77      0.83       559

    accuracy                           0.70      2721
   macro avg       0.72      0.70      0.69      2721
weighted avg       0.72      0.70      0.70      2721
86/86 [==============================] - 1s 9ms/step
Evaluación sobre datos reconstruidos: GAF
              precision    recall  f1-score   support

         0.0       0.78      0.86      0.82       543
         1.0       0.99      0.99      0.99       532
         2.0       0.84      0.71      0.77       536
         3.0       0.87      0.85      0.86       551
         4.0       0.83      0.87      0.85       559

    accuracy                           0.86      2721
   macro avg       0.86      0.86      0.86      2721
weighted avg       0.86      0.86      0.86      2721

RANDOM FOREST
Evaluación sobre validation_data.npy: ORIGINAL
              precision    recall  f1-score   support

         0.0       0.84      0.81      0.82       543
         1.0       0.94      0.99      0.96       532
         2.0       0.78      0.80      0.79       536
         3.0       0.90      0.94      0.92       551
         4.0       0.96      0.88      0.92       559

    accuracy                           0.88      2721
   macro avg       0.88      0.88      0.88      2721
weighted avg       0.89      0.88      0.88      2721

Evaluación sobre datos reconstruidos: RP
              precision    recall  f1-score   support

         0.0       0.69      0.19      0.30       543
         1.0       0.70      0.86      0.77       532
         2.0       0.45      0.70      0.54       536
         3.0       0.80      0.84      0.82       551
         4.0       0.83      0.74      0.78       559

    accuracy                           0.67      2721
   macro avg       0.69      0.67      0.64      2721
weighted avg       0.69      0.67      0.64      2721

Evaluación sobre datos reconstruidos: MTF
              precision    recall  f1-score   support

         0.0       0.57      0.35      0.43       543
         1.0       0.55      1.00      0.71       532
         2.0       0.48      0.46      0.47       536
         3.0       0.94      0.84      0.89       551
         4.0       0.95      0.71      0.81       559

    accuracy                           0.67      2721
   macro avg       0.70      0.67      0.66      2721
weighted avg       0.70      0.67      0.67      2721

Evaluación sobre datos reconstruidos: GAF
              precision    recall  f1-score   support

         0.0       0.84      0.80      0.82       543
         1.0       0.94      0.99      0.96       532
         2.0       0.77      0.81      0.79       536
         3.0       0.91      0.94      0.92       551
         4.0       0.96      0.87      0.92       559

    accuracy                           0.88      2721
   macro avg       0.88      0.88      0.88      2721
weighted avg       0.88      0.88      0.88      2721


Support vector machine


Evaluación sobre validation_data.npy:ORGINAL
              precision    recall  f1-score   support

         0.0       0.76      0.65      0.70       543
         1.0       0.97      0.98      0.97       532
         2.0       0.65      0.70      0.67       536
         3.0       0.82      0.84      0.83       551
         4.0       0.79      0.82      0.80       559

    accuracy                           0.80      2721
   macro avg       0.80      0.80      0.80      2721
weighted avg       0.80      0.80      0.80      2721

Evaluación sobre datos reconstruidos:RP 
              precision    recall  f1-score   support

         0.0       0.57      0.11      0.19       543
         1.0       0.75      0.89      0.82       532
         2.0       0.45      0.33      0.38       536
         3.0       0.72      0.76      0.74       551
         4.0       0.43      0.77      0.55       559

    accuracy                           0.57      2721
   macro avg       0.58      0.57      0.53      2721
weighted avg       0.58      0.57      0.53      2721



Evaluación sobre datos reconstruidos: MTF
              precision    recall  f1-score   support

         0.0       0.57      0.62      0.59       543
         1.0       0.53      1.00      0.69       532
         2.0       0.39      0.11      0.17       536
         3.0       0.85      0.77      0.81       551
         4.0       0.87      0.73      0.79       559

    accuracy                           0.65      2721
   macro avg       0.64      0.65      0.61      2721
weighted avg       0.64      0.65      0.61      2721

Evaluación sobre datos reconstruidos: GAF
              precision    recall  f1-score   support

         0.0       0.76      0.65      0.70       543
         1.0       0.97      0.98      0.97       532
         2.0       0.65      0.69      0.67       536
         3.0       0.82      0.84      0.83       551
         4.0       0.78      0.82      0.80       559

    accuracy                           0.80      2721
   macro avg       0.80      0.80      0.80      2721
weighted avg       0.80      0.80      0.80      2721

\section{Reconstructions from the Artificial Time Series}

For our first experiment, we construct a synthetic time series \( \mathbf{w} \in \mathbb{R}^{N \times 3} \) of length \(N=129\) by sampling an underlying parameter \(t\) uniformly over \([0,\pi]\).  Concretely, define
\[
t_i = \frac{i}{N-1}\,\pi,\qquad i=0,1,\dots,N-1,
\]
and let each vector \(\mathbf{w}_i = [w_{i,0},\,w_{i,1},\,w_{i,2}]^\top\) be given by
\[
w_{i,0} = \cos(t_i), 
\qquad
w_{i,1} = \sin(t_i),
\qquad
w_{i,2} = \sin(t_i)\;-\;\cos(t_i).
\]
Thus the full data matrix is
\[
\mathbf{w} = 
\begin{bmatrix}
\cos(t_0) & \sin(t_0) & \sin(t_0)-\cos(t_0) \\
\cos(t_1) & \sin(t_1) & \sin(t_1)-\cos(t_1) \\
\vdots    & \vdots    & \vdots             \\
\cos(t_{N-1}) & \sin(t_{N-1}) & \sin(t_{N-1})-\cos(t_{N-1})
\end{bmatrix}.
\]
This three‐dimensional signal—composed of a pure cosine channel, a pure sine channel, and their difference serves as the input for our RP, GAF and MTF reconstructions.
Figure \ref{fig:artificial_encodings} shows the image encodings of the three  channels under each technique:

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth,height=0.13\textheight]{images/COSRP.png}
        \caption{ RP}
        \label{fig:cos_rp}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth,height=0.13\textheight]{images/COSGasf.png}
        \caption{GASF}
        \label{fig:cos_gasf}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \includegraphics[width=\linewidth,height=0.13\textheight]{images/COSmtf.png}
        \caption{MTF}
        \label{fig:cos_mtf}
    \end{subfigure}
    \caption{Image encodings of the cosine channel from the artificial time series.}
    \label{fig:artificial_encodings}
\end{figure}
Once created the images we calculate the reconstruction through the decoding of the images:let's see the graphic comparison. 
\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.31\textheight]{images/Comparison_all_wideRP.png}
    \caption{Comparasion RP reconstruction with Artifical time series.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.31\textheight]{images/Comparison_all_wideGAF.png}
    \caption{Comparasion from GASF reconstruction with Artifical time series.}
\end{figure}

\begin{figure}[H]
    \centering
    \includegraphics[width=\linewidth,height=0.31\textheight]{images/Comparison_all_widemtf.png}
    \caption{Comparasion from MTF reconstruction with Artifical time series.}
\end{figure}


GADF

ERROR TIPO  Error Absoluto Promedio en dim 0 es : 1.4758660362371547
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 0 es : 1.4729683386262715
ERROR TIPO  Error Absoluto Promedio en dim 1 es : 1.4408761423889742
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 1 es : 1.611581967049091
ERROR TIPO  Error Absoluto Promedio en dim 2 es : 1.488423176809045
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 2 es : 1.8459922177473689
ERROR TIPO  Error Absoluto Promedio  medio global es : 1.4683884518117247
Desviación típica de ERROR TIPO Error Absoluto Promedio es : 1.6508314736056933
ERROR TIPO  Error Relativo Promedio en dim 0 es : 1.0933976610520575
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 0 es : 6.112235817938316
ERROR TIPO  Error Relativo Promedio en dim 1 es : 2.1246114710876745
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 1 es : 9.476956713638167
ERROR TIPO  Error Relativo Promedio en dim 2 es : 3.459839711894094
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 2 es : 8.362207570750677
ERROR TIPO  Error Relativo Promedio  medio global es : 2.2235987239801083
Desviación típica de ERROR TIPO Error Relativo Promedio es : 8.162914184450106
ERROR TIPO  Error Quadrático medio en dim 0 es : 4.842046856287526
Desviación típica de ERROR TIPO Error Quadrático medio en dim 0 es : 9.841708207954476
ERROR TIPO  Error Quadrático medio en dim 1 es : 5.864080099543405
Desviación típica de ERROR TIPO Error Quadrático medio en dim 1 es : 11.926631852061247
ERROR TIPO  Error Quadrático medio en dim 2 es : 6.028738718226134
Desviación típica de ERROR TIPO Error Quadrático medio en dim 2 es : 16.779545614072166
ERROR TIPO  Error Quadrático medio  medio global es : 5.578288558019022
Desviación típica de ERROR TIPO Error Quadrático medio es : 13.18437904991174
ERROR TIPO  Error desviacion típica en dim 0 es : 0.5738499030474465
Desviación típica de ERROR TIPO Error desviacion típica en dim 0 es : 0.6099026169600191
ERROR TIPO  Error desviacion típica en dim 1 es : 0.6582355638842718
Desviación típica de ERROR TIPO Error desviacion típica en dim 1 es : 1.005866931152574
ERROR TIPO  Error desviacion típica en dim 2 es : 0.4686833846698505
Desviación típica de ERROR TIPO Error desviacion típica en dim 2 es : 0.590254071705429
ERROR TIPO  Error desviacion típica  medio global es : 0.5669229505338563
Desviación típica de ERROR TIPO Error desviacion típica es : 0.7638032780882285
ERROR TIPO  Coeficiente de correlación pearson en dim 0 es : 0.9832681489343044
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 0 es : 0.02796953792565408
ERROR TIPO  Coeficiente de correlación pearson en dim 1 es : 0.9847110166542307
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 1 es : 0.03208959552205759
ERROR TIPO  Coeficiente de correlación pearson en dim 2 es : 0.9816158474387754
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 2 es : 0.02432054776988834
ERROR TIPO  Coeficiente de correlación pearson  medio global es : 0.98319833767577
Desviación típica de ERROR TIPO Coeficiente de correlación pearson es : 0.028333275978521314
min r Promedio [] -0.6490756189136719
[]
[[6.39082646e-02 3.11757233e-01 2.43943222e-01]
 [1.09080264e-01 1.57812748e+01 2.49536634e-02]
 [4.23177705e-03 9.98707133e-02 6.08199737e-02]
 [1.45237024e-02 5.77336170e-02 4.82990930e-02]
 [9.93007182e-01 9.80162363e-01 8.57694145e-01]]
MEDIA Q 5.578288558019022 5.576731387432264 428
MEDIA P 0.98319833767577 0.9832007315932781 503
Min Q 2.756241349401522e-05 2.756241349401522e-05 4451
Min P 0.4313220838061853 0.4313220838061853 4495
Max Q 150.57688554440753 150.57688554440753 3659
Max P 0.9999299054793269 0.9999299054793269 267



env) PS C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales> python scripts\generate_errors.py
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning:

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP).

For more information see: https://github.com/tensorflow/addons/issues/2807

  warnings.warn(
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.12.0 and strictly below 2.15.0 (nightly versions are not supported).
 The versions of TensorFlow you are currently using is 2.11.0 and is not supported.
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version.
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
errores (5484, 5, 3)
ERROR TIPO  Error Absoluto Promedio en dim 0 es : 1.4758660362371547
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 0 es : 1.4729683386262715
ERROR TIPO  Error Absoluto Promedio en dim 1 es : 1.4408761423889742
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 1 es : 1.611581967049091
ERROR TIPO  Error Absoluto Promedio en dim 2 es : 1.488423176809045
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 2 es : 1.8459922177473689
ERROR TIPO  Error Absoluto Promedio  medio global es : 1.4683884518117247
Desviación típica de ERROR TIPO Error Absoluto Promedio es : 1.6508314736056933
ERROR TIPO  Error Relativo Promedio en dim 0 es : 1.0933976610520575
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 0 es : 6.112235817938316
ERROR TIPO  Error Relativo Promedio en dim 1 es : 2.1246114710876745
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 1 es : 9.476956713638167
ERROR TIPO  Error Relativo Promedio en dim 2 es : 3.459839711894094
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 2 es : 8.362207570750677
ERROR TIPO  Error Relativo Promedio  medio global es : 2.2235987239801083
Desviación típica de ERROR TIPO Error Relativo Promedio es : 8.162914184450106
ERROR TIPO  Error Quadrático medio en dim 0 es : 4.842046856287526
Desviación típica de ERROR TIPO Error Quadrático medio en dim 0 es : 9.841708207954476
ERROR TIPO  Error Quadrático medio en dim 1 es : 5.864080099543405
Desviación típica de ERROR TIPO Error Quadrático medio en dim 1 es : 11.926631852061247
ERROR TIPO  Error Quadrático medio en dim 2 es : 6.028738718226134
Desviación típica de ERROR TIPO Error Quadrático medio en dim 2 es : 16.779545614072166
ERROR TIPO  Error Quadrático medio  medio global es : 5.578288558019022
Desviación típica de ERROR TIPO Error Quadrático medio es : 13.18437904991174
ERROR TIPO  Error desviacion típica en dim 0 es : 0.5738499030474465
Desviación típica de ERROR TIPO Error desviacion típica en dim 0 es : 0.6099026169600191
ERROR TIPO  Error desviacion típica en dim 1 es : 0.6582355638842718
Desviación típica de ERROR TIPO Error desviacion típica en dim 1 es : 1.005866931152574
ERROR TIPO  Error desviacion típica en dim 2 es : 0.4686833846698505
Desviación típica de ERROR TIPO Error desviacion típica en dim 2 es : 0.590254071705429
ERROR TIPO  Error desviacion típica  medio global es : 0.5669229505338563
Desviación típica de ERROR TIPO Error desviacion típica es : 0.7638032780882285
ERROR TIPO  Coeficiente de correlación pearson en dim 0 es : 0.9832681489343044
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 0 es : 0.02796953792565408
ERROR TIPO  Coeficiente de correlación pearson en dim 1 es : 0.9847110166542307
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 1 es : 0.03208959552205759
ERROR TIPO  Coeficiente de correlación pearson en dim 2 es : 0.9816158474387754
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 2 es : 0.02432054776988834
ERROR TIPO  Coeficiente de correlación pearson  medio global es : 0.98319833767577
Desviación típica de ERROR TIPO Coeficiente de correlación pearson es : 0.028333275978521314
min r Promedio [] -0.6490756189136719
[]
[[6.39082646e-02 3.11757233e-01 2.43943222e-01]
 [1.09080264e-01 1.57812748e+01 2.49536634e-02]
 [4.23177705e-03 9.98707133e-02 6.08199737e-02]
 [1.45237024e-02 5.77336170e-02 4.82990930e-02]
 [9.93007182e-01 9.80162363e-01 8.57694145e-01]]
MEDIA Q 5.578288558019022 5.576731387432264 428
MEDIA P 0.98319833767577 0.9832007315932781 503
Min Q 2.756241349401522e-05 2.756241349401522e-05 4451
Min P 0.4313220838061853 0.4313220838061853 4495
Max Q 150.57688554440753 150.57688554440753 3659
Max P 0.9999299054793269 0.9999299054793269 267
(venv) PS C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales> Remove-Item "test_error_debug.py"
(venv) PS C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales> python scripts\generate_errors.py
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning:

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP).

For more information see: https://github.com/tensorflow/addons/issues/2807

  warnings.warn(
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.12.0 and strictly below 2.15.0 (nightly versions are not supported).
 The versions of TensorFlow you are currently using is 2.11.0 and is not supported.
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version.
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
errores (5484, 5, 3)
ERROR TIPO  Error Absoluto Promedio en dim 0 es : 0.02448755598835313
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 0 es : 0.02598354278319255
ERROR TIPO  Error Absoluto Promedio en dim 1 es : 0.01978599041474392
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 1 es : 0.022699169351912667
ERROR TIPO  Error Absoluto Promedio en dim 2 es : 0.018598478849003657
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 2 es : 0.022284085204201373
ERROR TIPO  Error Absoluto Promedio  medio global es : 0.020957341750700232
Desviación típica de ERROR TIPO Error Absoluto Promedio es : 0.02384936022150902
ERROR TIPO  Error Relativo Promedio en dim 0 es : 0.01701005521824127
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 0 es : 0.07149781094766093
ERROR TIPO  Error Relativo Promedio en dim 1 es : 0.014419840255911887
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 1 es : 0.05141554498437939
ERROR TIPO  Error Relativo Promedio en dim 2 es : 0.04037428620751869
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 2 es : 0.17080657518631562
ERROR TIPO  Error Relativo Promedio  medio global es : 0.02390133378314893
Desviación típica de ERROR TIPO Error Relativo Promedio es : 0.11140859567315739
ERROR TIPO  Error Quadrático medio en dim 0 es : 0.005352643146814869
Desviación típica de ERROR TIPO Error Quadrático medio en dim 0 es : 0.012466457578518558
ERROR TIPO  Error Quadrático medio en dim 1 es : 0.0049690730265172615
Desviación típica de ERROR TIPO Error Quadrático medio en dim 1 es : 0.013222862045350553
ERROR TIPO  Error Quadrático medio en dim 2 es : 0.003206435179312385
Desviación típica de ERROR TIPO Error Quadrático medio en dim 2 es : 0.013355668344162679
ERROR TIPO  Error Quadrático medio  medio global es : 0.0045093837842148385
Desviación típica de ERROR TIPO Error Quadrático medio es : 0.013054381100684966
ERROR TIPO  Error desviacion típica en dim 0 es : 0.047970860940055406
Desviación típica de ERROR TIPO Error desviacion típica en dim 0 es : 0.054050004213703845
ERROR TIPO  Error desviacion típica en dim 1 es : 0.04176540734150999
Desviación típica de ERROR TIPO Error desviacion típica en dim 1 es : 0.056171141025189324
ERROR TIPO  Error desviacion típica en dim 2 es : 0.03324080881248565
Desviación típica de ERROR TIPO Error desviacion típica en dim 2 es : 0.045018971124346045
ERROR TIPO  Error desviacion típica  medio global es : 0.04099235903135035
Desviación típica de ERROR TIPO Error desviacion típica es : 0.05232173361055944
ERROR TIPO  Coeficiente de correlación pearson en dim 0 es : 0.9998665898410432
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 0 es : 0.00022367262433544967
ERROR TIPO  Coeficiente de correlación pearson en dim 1 es : 0.9998998767169788
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 1 es : 0.00017391067830814196
ERROR TIPO  Coeficiente de correlación pearson en dim 2 es : 0.9998694610300813
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 2 es : 0.00023919929564145806
ERROR TIPO  Coeficiente de correlación pearson  medio global es : 0.9998786425293679
Desviación típica de ERROR TIPO Coeficiente de correlación pearson es : 0.00021460904601202415
min r Promedio [] 0.9895301019536494
[]
[[5.88150482e-04 1.84959805e-03 1.34390698e-03]
 [9.38851832e-04 9.24642280e-02 1.37673772e-04]
 [5.62074455e-07 4.95051448e-06 2.38378357e-06]
 [7.46255686e-04 2.20422834e-03 1.54888098e-03]
 [9.99974794e-01 9.99951357e-01 9.99810506e-01]]
MEDIA Q 0.0045093837842148385 0.004508339001888782 3924
MEDIA P 0.9998786425293678 0.9998786471853119 854
Min Q 5.1590129842752255e-09 5.1590129842752255e-09 4451
Min P 0.996493302833391 0.996493302833391 338
Max Q 0.23931822667335154 0.23931822667335154 3690
Max P 0.999987677179346 0.999987677179346 4803
(venv) PS C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales> python scripts\generate_errors.py
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning:

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP).

For more information see: https://github.com/tensorflow/addons/issues/2807

  warnings.warn(
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.12.0 and strictly below 2.15.0 (nightly versions are not supported).
 The versions of TensorFlow you are currently using is 2.11.0 and is not supported.
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version.
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
errores (5484, 5, 3)
ERROR TIPO  Error Absoluto Promedio en dim 0 es : 5.355695938290749
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 0 es : 5.651871245010194
ERROR TIPO  Error Absoluto Promedio en dim 1 es : 5.518405859894898
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 1 es : 6.279573107411369
ERROR TIPO  Error Absoluto Promedio en dim 2 es : 4.650752782384816
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 2 es : 5.581959612277596
ERROR TIPO  Error Absoluto Promedio  medio global es : 5.1749515268568205
Desviación típica de ERROR TIPO Error Absoluto Promedio es : 5.858338041898022
ERROR TIPO  Error Relativo Promedio en dim 0 es : 3.5317178125144473
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 0 es : 15.86787946961436
ERROR TIPO  Error Relativo Promedio en dim 1 es : 6.04498293286396
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 1 es : 20.3114529272773
ERROR TIPO  Error Relativo Promedio en dim 2 es : 9.486131310819056
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 2 es : 25.291138043970935
ERROR TIPO  Error Relativo Promedio  medio global es : 6.348472559307763
Desviación típica de ERROR TIPO Error Relativo Promedio es : 20.981664504853576
ERROR TIPO  Error Quadrático medio en dim 0 es : 78.22399513550457
Desviación típica de ERROR TIPO Error Quadrático medio en dim 0 es : 134.5759737847093
ERROR TIPO  Error Quadrático medio en dim 1 es : 78.69683690740152
Desviación típica de ERROR TIPO Error Quadrático medio en dim 1 es : 139.91292313039054
ERROR TIPO  Error Quadrático medio en dim 2 es : 55.537150745782
Desviación típica de ERROR TIPO Error Quadrático medio en dim 2 es : 102.25225953526747
ERROR TIPO  Error Quadrático medio  medio global es : 70.82200844604894
Desviación típica de ERROR TIPO Error Quadrático medio es : 127.13542566520854
ERROR TIPO  Error desviacion típica en dim 0 es : 6.069405331067748
Desviación típica de ERROR TIPO Error desviacion típica en dim 0 es : 6.413072731784571
ERROR TIPO  Error desviacion típica en dim 1 es : 6.181306412450888
Desviación típica de ERROR TIPO Error desviacion típica en dim 1 es : 7.03394042868614
ERROR TIPO  Error desviacion típica en dim 2 es : 5.2374355320428645
Desviación típica de ERROR TIPO Error desviacion típica en dim 2 es : 6.331020492884792
ERROR TIPO  Error desviacion típica  medio global es : 5.829382425187166
Desviación típica de ERROR TIPO Error desviacion típica es : 6.613559377300888
ERROR TIPO  Coeficiente de correlación pearson en dim 0 es : 0.03410131208119769
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 0 es : 0.6201351724572863
ERROR TIPO  Coeficiente de correlación pearson en dim 1 es : 0.0005784367571269541
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 1 es : 0.6239768349142522
ERROR TIPO  Coeficiente de correlación pearson en dim 2 es : 0.0005396344911894362
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 2 es : 0.616659477532604
ERROR TIPO  Coeficiente de correlación pearson  medio global es : 0.01173979444317136
Desviación típica de ERROR TIPO Coeficiente de correlación pearson es : 0.6204658708901567
min r Promedio [] -0.9699289756902395
[]
[[ 2.64559290e-01  6.25260716e-01  2.87192145e-01]
 [ 4.57061477e-01  5.07968920e+01  2.94089183e-02]
 [ 1.05863916e-01  5.69497902e-01  1.33281081e-01]
 [ 3.25071083e-01  7.57613943e-01  3.32839097e-01]
 [-7.78644087e-01 -1.72914905e-01  3.16612644e-01]]
MEDIA Q 82.90956853136291 82.93066845534945 3266
MEDIA P 0.01173979444317136 0.011672747864274746 192
Min Q 0.00036289241837612055 0.00036289241837612055 4451
Min P -0.9503670767246145 -0.9503670767246145 5425
Max Q 3495.489604089574 3495.489604089574 3691
Max P 0.9431748694215948 0.9431748694215948 1688
(venv) PS C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales> python scripts\generate_errors.py
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning:

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP).

For more information see: https://github.com/tensorflow/addons/issues/2807

  warnings.warn(
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.12.0 and strictly below 2.15.0 (nightly versions are not supported).
 The versions of TensorFlow you are currently using is 2.11.0 and is not supported.
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version.
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
errores (5484, 5, 3)
ERROR TIPO  Error Absoluto Promedio en dim 0 es : 4.179893683619314
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 0 es : 4.005691907647848
ERROR TIPO  Error Absoluto Promedio en dim 1 es : 4.06277315770366
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 1 es : 4.236819241545243
ERROR TIPO  Error Absoluto Promedio en dim 2 es : 3.372748133013874
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 2 es : 3.733143344431444
ERROR TIPO  Error Absoluto Promedio  medio global es : 3.8718049914456163
Desviación típica de ERROR TIPO Error Absoluto Promedio es : 4.013020865686816
ERROR TIPO  Error Relativo Promedio en dim 0 es : 2.743856524810035
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 0 es : 18.371275327885712
ERROR TIPO  Error Relativo Promedio en dim 1 es : 3.636526795274954
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 1 es : 14.87487588242177
ERROR TIPO  Error Relativo Promedio en dim 2 es : 6.476515752157649
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 2 es : 16.63729797909807
ERROR TIPO  Error Relativo Promedio  medio global es : 4.281372982418879
Desviación típica de ERROR TIPO Error Relativo Promedio es : 16.76432747778026
ERROR TIPO  Error Quadrático medio en dim 0 es : 46.20943557424746
Desviación típica de ERROR TIPO Error Quadrático medio en dim 0 es : 77.19695165423927
ERROR TIPO  Error Quadrático medio en dim 1 es : 46.5579867126718
Desviación típica de ERROR TIPO Error Quadrático medio en dim 1 es : 89.51274518087898
ERROR TIPO  Error Quadrático medio en dim 2 es : 31.671343307124467
Desviación típica de ERROR TIPO Error Quadrático medio en dim 2 es : 67.88446227352952
ERROR TIPO  Error Quadrático medio  medio global es : 41.48487609768376
Desviación típica de ERROR TIPO Error Quadrático medio es : 79.00545971224844
ERROR TIPO  Error desviacion típica en dim 0 es : 3.9829708750358526
Desviación típica de ERROR TIPO Error desviacion típica en dim 0 es : 4.218807861471984
ERROR TIPO  Error desviacion típica en dim 1 es : 4.061693779115357
Desviación típica de ERROR TIPO Error desviacion típica en dim 1 es : 4.65391492550182
ERROR TIPO  Error desviacion típica en dim 2 es : 3.193462698873537
Desviación típica de ERROR TIPO Error desviacion típica en dim 2 es : 3.6872539179083237
ERROR TIPO  Error desviacion típica  medio global es : 3.7460424510082486
Desviación típica de ERROR TIPO Error desviacion típica es : 4.2235143574387575
ERROR TIPO  Coeficiente de correlación pearson en dim 0 es : 0.30622383689350036
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 0 es : 0.4367135782644822
ERROR TIPO  Coeficiente de correlación pearson en dim 1 es : 0.2834414938048701
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 1 es : 0.4199883552033051
ERROR TIPO  Coeficiente de correlación pearson en dim 2 es : 0.2936849847181679
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 2 es : 0.4162839853769611
ERROR TIPO  Coeficiente de correlación pearson  medio global es : 0.2944501051388461
Desviación típica de ERROR TIPO Coeficiente de correlación pearson es : 0.4245239373386236
min r Promedio [] -0.896310755215644
[]
[[ 4.81845191e-01  1.23613522e+00  1.21064521e+00]
 [ 7.95820611e-01  3.63813617e+01  1.23841963e-01]
 [ 2.96930847e-01  2.19514446e+00  1.62410728e+00]
 [ 3.79549469e-01  1.48546031e+00  9.91432299e-01]
 [-2.06618655e-02  2.18273067e-01 -4.48105098e-03]]
MEDIA Q 43.32549897787611 43.29265726666903 1888
MEDIA P 0.2944501051388461 0.2944144714625078 1696
Min Q 0.00015538287517934295 0.00015538287517934295 4464
Min P -0.6327324279467877 -0.6327324279467877 293
Max Q 1637.7044625780866 1637.7044625780866 3691
Max P 0.9619914216992242 0.9619914216992242 1867
(venv) PS C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales> python scripts\generate_errors.py
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning:

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP).

For more information see: https://github.com/tensorflow/addons/issues/2807

  warnings.warn(
C:\Users\adria\OneDrive\Documentos\TFG-Tratamiento-de-Series-Temporales\venv\lib\site-packages\tensorflow_addons\utils\ensure_tf_install.py:53: UserWarning: Tensorflow Addons supports using Python ops for all Tensorflow versions above or equal to 2.12.0 and strictly below 2.15.0 (nightly versions are not supported).
 The versions of TensorFlow you are currently using is 2.11.0 and is not supported.
Some things might work, some things might not.
If you were to encounter a bug, do not file an issue.
If you want to make sure you're using a tested and supported configuration, either change the TensorFlow version or the TensorFlow Addons's version.
You can find the compatibility matrix in TensorFlow Addon's readme:
https://github.com/tensorflow/addons
  warnings.warn(
errores (5484, 5, 3)
ERROR TIPO  Error Absoluto Promedio en dim 0 es : 1.4758660362371547
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 0 es : 1.4729683386262715
ERROR TIPO  Error Absoluto Promedio en dim 1 es : 1.4408761423889742
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 1 es : 1.611581967049091
ERROR TIPO  Error Absoluto Promedio en dim 2 es : 1.488423176809045
Desviación típica de ERROR TIPO Error Absoluto Promedio en dim 2 es : 1.8459922177473689
ERROR TIPO  Error Absoluto Promedio  medio global es : 1.4683884518117247
Desviación típica de ERROR TIPO Error Absoluto Promedio es : 1.6508314736056933
ERROR TIPO  Error Relativo Promedio en dim 0 es : 1.0933976610520575
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 0 es : 6.112235817938316
ERROR TIPO  Error Relativo Promedio en dim 1 es : 2.1246114710876745
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 1 es : 9.476956713638167
ERROR TIPO  Error Relativo Promedio en dim 2 es : 3.459839711894094
Desviación típica de ERROR TIPO Error Relativo Promedio en dim 2 es : 8.362207570750677
ERROR TIPO  Error Relativo Promedio  medio global es : 2.2235987239801083
Desviación típica de ERROR TIPO Error Relativo Promedio es : 8.162914184450106
ERROR TIPO  Error Quadrático medio en dim 0 es : 4.842046856287526
Desviación típica de ERROR TIPO Error Quadrático medio en dim 0 es : 9.841708207954476
ERROR TIPO  Error Quadrático medio en dim 1 es : 5.864080099543405
Desviación típica de ERROR TIPO Error Quadrático medio en dim 1 es : 11.926631852061247
ERROR TIPO  Error Quadrático medio en dim 2 es : 6.028738718226134
Desviación típica de ERROR TIPO Error Quadrático medio en dim 2 es : 16.779545614072166
ERROR TIPO  Error Quadrático medio  medio global es : 5.578288558019022
Desviación típica de ERROR TIPO Error Quadrático medio es : 13.18437904991174
ERROR TIPO  Error desviacion típica en dim 0 es : 0.5738499030474465
Desviación típica de ERROR TIPO Error desviacion típica en dim 0 es : 0.6099026169600191
ERROR TIPO  Error desviacion típica en dim 1 es : 0.6582355638842718
Desviación típica de ERROR TIPO Error desviacion típica en dim 1 es : 1.005866931152574
ERROR TIPO  Error desviacion típica en dim 2 es : 0.4686833846698505
Desviación típica de ERROR TIPO Error desviacion típica en dim 2 es : 0.590254071705429
ERROR TIPO  Error desviacion típica  medio global es : 0.5669229505338563
Desviación típica de ERROR TIPO Error desviacion típica es : 0.7638032780882285
ERROR TIPO  Coeficiente de correlación pearson en dim 0 es : 0.9832681489343044
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 0 es : 0.02796953792565408
ERROR TIPO  Coeficiente de correlación pearson en dim 1 es : 0.9847110166542307
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 1 es : 0.03208959552205759
ERROR TIPO  Coeficiente de correlación pearson en dim 2 es : 0.9816158474387754
Desviación típica de ERROR TIPO Coeficiente de correlación pearson en dim 2 es : 0.02432054776988834
ERROR TIPO  Coeficiente de correlación pearson  medio global es : 0.98319833767577
Desviación típica de ERROR TIPO Coeficiente de correlación pearson es : 0.028333275978521314
min r Promedio [] -0.6490756189136719
[]
[[6.39082646e-02 3.11757233e-01 2.43943222e-01]
 [1.09080264e-01 1.57812748e+01 2.49536634e-02]
 [4.23177705e-03 9.98707133e-02 6.08199737e-02]
 [1.45237024e-02 5.77336170e-02 4.82990930e-02]
 [9.93007182e-01 9.80162363e-01 8.57694145e-01]]
MEDIA Q 5.578288558019022 5.576731387432264 428
MEDIA P 0.98319833767577 0.9832007315932781 503
Min Q 2.756241349401522e-05 2.756241349401522e-05 4451
Min P 0.4313220838061853 0.4313220838061853 4495
Max Q 150.57688554440753 150.57688554440753 3659
Max P 0.9999299054793269 0.9999299054793269 267
(venv) PS C:\Users\adria\OneDrive\Documentos\TFG-