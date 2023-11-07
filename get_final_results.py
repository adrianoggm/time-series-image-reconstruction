#!/home/fmgarmor/miot_env/bin/python3

import os
import numpy as np
import pandas as pd


base_dir = 'results/evaluation_synthetic_quality'

def metrics(confusion_matrix):
    true_positive = np.diag(confusion_matrix)
    accuracy = np.sum(true_positive) / np.sum(confusion_matrix)
    
    precision = np.mean([true_positive[i] / np.sum(confusion_matrix[:, i]) for i in range(confusion_matrix.shape[1]) if np.sum(confusion_matrix[:, i]) != 0])
    
    recall = np.mean([true_positive[i] / np.sum(confusion_matrix[i, :]) for i in range(confusion_matrix.shape[0]) if np.sum(confusion_matrix[i, :]) != 0])
    
    true_negative = np.sum(confusion_matrix) - (np.sum(confusion_matrix, axis=0) + np.sum(confusion_matrix, axis=1) - true_positive)
    false_positive = np.sum(confusion_matrix, axis=0) - true_positive
    specificity = np.mean(true_negative / (true_negative + false_positive))
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, specificity, f1_score


def process_fold_for_model(model, fold_dir):
    with open(os.path.join(fold_dir, model, f'cm_fold_{fold_dir[-1]}.txt'), 'r') as f:
        matrix = np.array([list(map(int, line.strip().split())) for line in f])
        return metrics(matrix)

def process_sampling_epochs(sampling, prefix_folder, synthetic_data=True):
    p = 'result_models_trained_on_synthetic_images' if synthetic_data else "result_models_trained_on_real_images"
    path = os.path.join(base_dir, sampling, prefix_folder, p)
    
    fold_dirs = [os.path.join(path, d) for d in os.listdir(path) if 'fold_' in d and os.path.isdir(os.path.join(path, d))]
    
    # Asumiendo que todos los folds tienen los mismos modelos, tomamos los modelos del primer fold
    models = [d for d in os.listdir(fold_dirs[0]) if os.path.isdir(os.path.join(fold_dirs[0], d))]

    results = []

    for model in models:
        accs, precisions, recalls, specificities, f1_scores = [], [], [], [], []
        
        for fold_dir in fold_dirs:
            accuracy, precision, recall, specificity, f1_score = process_fold_for_model(model, fold_dir)
            accs.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            specificities.append(specificity)
            f1_scores.append(f1_score)

        avg_acc, std_acc, avg_precision, std_precision, avg_recall, std_recall, avg_specificity, std_specificity, avg_f1, std_f1 = np.mean(accs), np.std(accs), np.mean(precisions), np.std(precisions), np.mean(recalls), np.std(recalls), np.mean(specificities), np.std(specificities), np.mean(f1_scores), np.std(f1_scores)
        results.append([model, avg_acc, std_acc, avg_f1, std_f1, avg_precision, std_precision, avg_recall, std_recall, avg_specificity, std_specificity])
    
    df = pd.DataFrame(results, columns=['Model', 'Avg Accuracy', 'Std Accuracy', 'Avg F1-Score', 'Std F1-Score', 'Avg Precision', 'Std Precision', 'Avg Recall', 'Std Recall', 'Avg Specificity', 'Std Specificity'])
    
    # Redondear las métricas
    metrics_cols = ['Avg Accuracy', 'Avg F1-Score', 'Avg Precision', 'Avg Recall', 'Avg Specificity']
    std_cols = ['Std Accuracy', 'Std F1-Score', 'Std Precision', 'Std Recall', 'Std Specificity']
    df[metrics_cols] = df[metrics_cols].round(4)
    df[std_cols] = df[std_cols].round(4)

    df.to_csv(os.path.join(path, f'{sampling}_{prefix_folder}_performace_average_results.csv'), index=False)
    df.to_excel(os.path.join(path, f'{sampling}_{prefix_folder}_performace_average_results.xlsx'), index=False, engine='openpyxl')  # Línea para exportar a Excel



for epochs in range(1000, 11000, 1000):
    print("epochs:", epochs)
    process_sampling_epochs('loto', f'epochs-{epochs}')



for epochs in range(1000, 11000, 1000):
    print("epochs:", epochs)
    process_sampling_epochs('loso', f'epochs-{epochs}')

process_sampling_epochs('loso', 'real-train-data', synthetic_data=False)
process_sampling_epochs('loto', 'real-train-data', synthetic_data=False)

