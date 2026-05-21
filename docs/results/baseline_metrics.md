# Baseline classification metrics

These are reference classification reports obtained on the WISDM validation set
during the original development of the repo. They are kept here as a frozen
snapshot — they are NOT regenerated automatically and may drift from current
code behaviour as encoders, reconstructors, or models evolve.

Numbers below cover four reconstruction sources (original windows, RP⁻¹, MTF⁻¹,
GASF⁻¹) evaluated by four classifiers (CNN, kNN, LSTM, Random Forest, SVC).

---

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

