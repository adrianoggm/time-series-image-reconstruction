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

