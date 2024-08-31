#!/home/adriano/Escritorio/TFG/venv/bin/python3

import os
import numpy as np
import pandas as pd
from functools import partial
import argparse
from tqdm import trange, tqdm
from torchvision import datasets, transforms, utils
from datasets import load_dataset
import keras
import k_diffusion as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from tgen import eval, utils


def compute_metrics_for_each_class(cm):
    num_classes = cm.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        # Treat current class as positive and all others as negative
        tp = cm[i, i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        # Compute metrics for this class
        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        specificity = tn / (tn + fp) if tn + fp != 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0

        metrics[f"class_{i}_precision"] = precision
        metrics[f"class_{i}_recall"] = recall
        metrics[f"class_{i}_f1"] = f1
        metrics[f"class_{i}_specificity"] = specificity
        metrics[f"class_{i}_accuracy"] = accuracy

    return metrics


def save_confusion_matrix(cm, path, normalized=False):
    """Save the confusion matrix to a txt file and as an image."""
    # Save to txt
    with open(path + ".txt", 'w') as f:
        for line in cm:
            f.write("\t".join(str(x) for x in line))
            f.write('\n')
    
    # Save as image
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    
    plt.figure(figsize=(7.5, 5.5))
    sns.heatmap(cm, annot=True, cmap="Blues")
    plt.savefig(path + ".png")


def main():
    '''Examples of runs:

    ** LOSO approach
    - train with synthetic images
    $  nohup ./eval_diffusion.py --model-name resnet50 --prefix exp-classes-all-classes --epochs 100 --synth-train --config configs/config_wisdm_128x128_loso.json > results/evaluation_synthetic_quality/loso/exp-classes-all-classes/eval.log &
    $  nohup ./eval_diffusion.py --model-name resnet50 --prefix exp-classes-3-4 --epochs 100 --synth-train > results/evaluation_synthetic_quality/exp-classes-3-4/eval.log &
    
    - train with real images
    $ nohup ./eval_diffusion.py --model-name resnet34 --prefix exp-classes-all-classes --epochs 30 > results/evaluation_synthetic_quality/eval-real-data.log &
    $ nohup ./eval_diffusion.py --model-name resnet50 --prefix exp-classes-3-4 --epochs 100 > results/evaluation_synthetic_quality/exp-classes-3-4/eval.log &

    ** LOTO approach
    - train with synthetic images
    $  nohup ./eval_diffusion.py --model-name vgg16 --prefix exp-classes-all-classes --epochs 30 --synth-train --config configs/config_wisdm_128x128_loto.json > results/evaluation_synthetic_quality/loso/exp-classes-all-classes/eval-vgg16.log &
    - train with real images
    $ nohup ./eval_diffusion.py --model-name resnet34 --prefix exp-classes-all-classes --epochs 30 > results/evaluation_synthetic_quality/eval-real-data.log &

    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')

    p.add_argument('--model-name', type=str, default="lstm-cnn", #vgg16
                   help='the model name') #models: https://github.com/qubvel/classification_models    
                   #"resnet18", "resnet50", "densenet121", "mobilenet", "simple-cnn"
    p.add_argument('--synth-train', action="store_true",
                   help='use synthetic images for train; if not, use real images') 
    p.add_argument('--prefered-device', type=int, default=-1, help='the prefered device to run the model')    
    p.add_argument('--img-batch-size', type=int, default=16,
                   help='the batch size for loading the images')         
    p.add_argument('--batch-size', type=int, default=16,
                   help='the batch size for the neural network')
    p.add_argument('--image-size', type=int, default=128,
                   help='the image resolution size')
    p.add_argument('--n-folds', type=int, default=3,
                   help='the number of folds')
    p.add_argument('--prefix', type=str, default='exp-classes-all-classes',
                   help='the output prefix')
    p.add_argument('--epochs', type=int, default=100,
                   help='the number of epochs to train de image classifier')
    p.add_argument('--val-split', type=float, default=0.0,
                   help='if we use a validation split, indicate a value; if not, 0.0')
    p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",help='the configuration file')

    args = p.parse_args()

    utils.set_gpu(args.gpu_id)
    
    config = K.config.load_config(open(args.config))
    dataset_config = config['dataset']
    sampling_method = dataset_config["sampling"]

    use_synth_images_for_train = args.synth_train
    print("use synthetic images for train?: ", use_synth_images_for_train)
    val_split = args.val_split
    model_name = args.model_name
    img_size = args.image_size
    epochs = args.epochs
    batch_size = args.batch_size
    img_batch_size = args.img_batch_size
    prefered_device = args.prefered_device
    # config = K.config.load_config(open(args.config))
    # model_config = config['model']
    # dataset_config = config['dataset']
    # size = model_config['input_size']
    # tf = transforms.Compose([
    #     transforms.Resize(size[0], interpolation=transforms.InterpolationMode.LANCZOS),
    #     transforms.CenterCrop(size[0]),
    #     K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob']),
    # ])

    # test_set = load_dataset(f"{dataset_config['location']}{args.current_fold}", split="test")
    # test_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key']))
    # print(test_set)




    # CLASS_NAMES = ["Walking", "Jogging", "Stairs", "Sitting", "Standing"]
    VALIDATION_ACCURACY = []
    VALIDATION_LOSS = []


    models, histories = [], []
    acc_per_class = {}

    dir_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/result_models_trained_on_real_images/" 
    if use_synth_images_for_train:
        dir_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/result_models_trained_on_synthetic_images/" 
    os.makedirs(dir_path, exist_ok=True)
    # release GPU Memory
    # K.clear_session()
    if val_split > 0:
        train_idg = ImageDataGenerator(rescale=1./255,  validation_split=val_split)
    else:
        train_idg = ImageDataGenerator(rescale=1./255)

    val_idg = ImageDataGenerator(rescale=1./255)
    test_idg = ImageDataGenerator(rescale=1./255)

    all_predicted_class_indices = []
    all_precisions = []
    all_recalls = []
    all_specificities = []
    all_accs = []
    all_f1s = []
    all_metrics_per_fold = []


    for fold in range(args.n_folds):
        train_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/"
        train_path = f"{train_path}train/" if use_synth_images_for_train else f"{train_path}real_train/"
        test_path = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/test/"
        val_path = test_path #f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/validation/"
        train_data = pd.read_csv(f'{train_path}{"" if use_synth_images_for_train else "real_"}training_labels.csv', dtype=str)[["filename", "label"]]
        test_data = pd.read_csv(f'{test_path}test_labels.csv', dtype=str)[["filename", "label"]]
        val_data = test_data#pd.read_csv(f'{val_path}val_labels.csv', dtype=str)[["filename", "label"]]
        # print("train path: ", train_path)
        # print("train shape: ", train_data.shape)
        # print("val path: ", val_path)
        # print("val shape: ", val_data.shape)
        # print("test path: ", test_path)
        # print("test shape: ", test_data.shape)

        print("Loading images")
        # from tgen.preprocess_crop import preprocess_crop
        #training data are the synthetic images
        
        if val_split < 0:
            train_data_generator = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                            x_col = "filename", y_col = "label",
                            # class_mode = "raw", 
                            class_mode="categorical",
                            shuffle = True, seed=33
                            )
            valid_data_generator = None
        elif val_split > 0:
            train_data_generator = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                        subset='training',
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )
        # for training - we used a 20% of train data to validate the model during training phase             
            valid_data_generator  = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                        subset='validation',
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )
        else:
            train_data_generator = train_idg.flow_from_dataframe(train_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = train_path,
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )
            valid_data_generator  = val_idg.flow_from_dataframe(val_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = val_path,
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = True, seed=33
                        )

        #to test the trained model, we used real test data unseen on training phase
        test_data_generator  = test_idg.flow_from_dataframe(test_data, batch_size=img_batch_size, target_size=(img_size, img_size), directory = test_path,
                    x_col = "filename", y_col = "label",
                    # class_mode = "raw", 
                    class_mode="categorical",
                    shuffle = False, #important shuffle=False to compare the prediction result
                    ) 
        # print("train_data_generator.class_indices", train_data_generator.class_indices)
        # print("train_data_generator.labels", train_data_generator.labels)
        # print("valid_data_generator.class_indices", valid_data_generator.class_indices)
        # print("valid_data_generator.labels", valid_data_generator.labels)
        # print("test_data_generator.class_indices", test_data_generator.class_indices)
        # print("test_data_generator.labels", test_data_generator.labels)
        
        # STEP_SIZE_TRAIN=train_data_generator.n // train_data_generator.batch_size
        # STEP_SIZE_VALID=valid_data_generator.n // valid_data_generator.batch_size
        # STEP_SIZE_TEST=test_data_generator.n // test_data_generator.batch_size
        STEP_SIZE_TEST = np.ceil(test_data_generator.n / test_data_generator.batch_size)


        train_labels = (train_data_generator.class_indices)
        print("Class original indices:", train_labels, "length:", len(train_labels))
        print("Class indices:", test_data_generator.class_indices, "length:", len(test_data_generator.class_indices))
        print("Class original labels:", train_data_generator.labels, "length:", len(train_data_generator.labels))

        train_labels = dict((v,k) for k,v in train_labels.items())
        print("Class indices using dict((v,k) for k,v in train_labels.items()):", train_labels, "length:", len(train_labels))

        n_classes =  len(train_labels)

        print("Number of classes:", n_classes)
        print("Building model:", model_name)

        #Train the model
        if model_name == "simple":
            model, history = eval.build_model_A(dir_path, model_name, fold, n_classes, (img_size, img_size, 3), train_data_generator, valid_data_generator, epochs, batch_size, prefered_device) #train2_ds, val2_ds, EPOCHS)
        elif model_name == "lstm-cnn":
            model, history = eval.generate_lstmfcn_xia(dir_path, model_name, fold, n_classes, (img_size, img_size, 3), train_data_generator, valid_data_generator, epochs, batch_size, prefered_device)
        else:
            model, history = eval.build_X(dir_path, model_name, fold, n_classes,  (img_size, img_size, 3), train_data_generator, valid_data_generator, epochs, batch_size, prefered_device) #train2_ds, val2_ds, EPOCHS)
        
        print("model built")
        models.append(model)
        histories.append(history)

        #plot history
        # eval.plot_train_eval(history, f"{p}% Synthetic Images") 

        # LOAD BEST MODEL to evaluate the performance of the model
        # model.load_weights(f"{dir_path}fold_{fold}/{model_name}/model_{model_name}")
        
        # results = model.evaluate(valid_data_generator)
        # results = model.evaluate(valid_data_generator, steps=STEP_SIZE_TEST)

        results = model.evaluate(test_data_generator) #, steps=STEP_SIZE_TEST, verbose=0)
        results = dict(zip(model.metrics_names,results))
        print("Results - validation:", results)
        
        VALIDATION_ACCURACY.append(results['accuracy'])
        VALIDATION_LOSS.append(results['loss'])

        
  
        
        #confusion matrix of current fold
        # y_pred = model.predict(test_data_generator)
        test_data_generator.reset() #You need to reset the test_generator before whenever you call the predict_generator. This is important, if you forget to reset the test_generator you will get outputs in a weird order.
        #ENSURE SAME LENGTH OF TRUE TEST_LABELS AND PREDICTIONS
        y_pred = model.predict(test_data_generator, steps=STEP_SIZE_TEST)
        print("steps",STEP_SIZE_TEST)
        print("y_pred length",len(y_pred))



        # print("y_pred:", y_pred)
        predicted_class_indices = np.argmax(y_pred,axis=1)
        all_predicted_class_indices.append(predicted_class_indices)

        for i,predicted_class_indices in enumerate(all_predicted_class_indices):
            print("fold", i, "length", len(test_data_generator.labels), "true_class_indices:", test_data_generator.labels)
            print("fold", i, "length", len(predicted_class_indices), "predicted_class_indices:", predicted_class_indices)
       
       # predicted_class_indices_axis__1 = np.argmax(y_pred,axis=-1)
        # print("predicted_class_indices_axis_-1:", predicted_class_indices_axis__1)

        #now predicted_class_indices has the predicted labels, but you can’t simply tell what the predictions are, because all you can see is numbers like 0,1,4,1,0,6…
        #and most importantly you need to map the predicted labels with their unique ids such as filenames to find out what you predicted for which image.
        
        # print("original labels:", labels)
        test_labels = (test_data_generator.class_indices)
        print("test indices true:", test_labels)
        print("test labels true:", test_data_generator.labels)

        test_labels = dict((v,k) for k,v in test_labels.items())
        print("test labels true using  test_labels.items():", test_labels)

        predictions = [test_labels[k] for k in predicted_class_indices]
        print("predictions:", predictions)
        
        # cm = confusion_matrix(test_data_generator.labels, predicted_class_indices)
        # save_confusion_matrix(cm, f"{dir_path}fold_{fold}/{model_name}/cm_fold_{fold}")
        # metrics_values = compute_metrics_for_each_class(cm)
        # acc_per_class[f"Metrics [fold-{fold}]"] = metrics_values
        cm = confusion_matrix(test_data_generator.labels, predicted_class_indices)
        save_confusion_matrix(cm, f"{dir_path}fold_{fold}/{model_name}/cm_fold_{fold}")
        metrics_values = compute_metrics_for_each_class(cm)
        acc_per_class[f"Metrics [fold-{fold}]"] = metrics_values
        
        all_metrics_per_fold.append(metrics_values)


        # Agregar a las listas de métricas globales
        all_precisions.extend([metrics_values[key] for key in metrics_values if "precision" in key])
        all_recalls.extend([metrics_values[key] for key in metrics_values if "recall" in key])
        all_specificities.extend([metrics_values[key] for key in metrics_values if "specificity" in key])
        all_accs.extend([metrics_values[key] for key in metrics_values if "accuracy" in key])
        all_f1s.extend([metrics_values[key] for key in metrics_values if "f1" in key])


        keras.backend.clear_session()



        # Finally, save the results to a CSV file.
        # print("Saving predictions")
        # filenames=test_data_generator.filenames
        # print("filenames:", len(filenames))
        # print("predictions:", len(predictions))
        # results=pd.DataFrame({"Filename": filenames, "Predictions": predictions})
        # os.makedirs(f"{dir_path}fold_{fold}/{model_name}/", exist_ok=True)
        # results.to_csv(f"{dir_path}fold_{fold}/{model_name}/prediction_results.csv",index=False)

        # # y_pred = np.argmax(y_pred, axis=1) #-1) 
        # metrics = eval.plot_cm(
        #     f'{dir_path}fold_{fold}/{model_name}/',
        #     model_name,
        #     test_data_generator.labels,
        #     predicted_class_indices,
        #     train_labels.values(),
        #     normalized=True,
        #     figsize=(12, 8),
        #     suffix_name=f"_fold_{fold}"
        #     )
        
     
        
        # acc_per_class[f"Confussion Matrix metrics [fold-{fold}]"] = metrics
        acc_per_class[f"Average Accuracy [fold-{fold}]"] = [VALIDATION_ACCURACY[fold]]
        acc_per_class[f"Average Loss [fold-{fold}]"] = [VALIDATION_LOSS[fold]]

        # tf.keras.backend.clear_session()
    # acc_per_class[f"Average Accuracy"] = [np.mean(np.array(VALIDATION_ACCURACY))]
    # acc_per_class[f"Average Loss"] = [np.mean(np.array(VALIDATION_LOSS))]

    # print(acc_per_class)
    

    # avg_accuracies_df = pd.DataFrame(acc_per_class)
    # os.makedirs(f"{dir_path}average/{model_name}/", exist_ok=True)
    # avg_accuracies_df.to_csv(f"{dir_path}average/{model_name}/test_accuracies.csv",index=False)

    #plot only one, since all of them share the same architecture
    # keras.utils.plot_model(model, f"{dir_path}average/{model_name}/model_multi_input_and_output_model.png", show_shapes=True)




    # avg_cm = np.mean([confusion_matrix(test_data_generator.labels, pc) for pc in all_predicted_class_indices], axis=0)
    # save_confusion_matrix(avg_cm, f"{dir_path}average/{model_name}/avg_cm", normalized=True)


    #---SAVE RESULTS
    # Reshape para que cada fila represente un fold y cada columna una clase
    all_precisions_matrix = np.array(all_precisions).reshape(args.n_folds, n_classes)
    all_recalls_matrix = np.array(all_recalls).reshape(args.n_folds, n_classes)
    all_specificities_matrix = np.array(all_specificities).reshape(args.n_folds, n_classes)
    all_accs_matrix = np.array(all_accs).reshape(args.n_folds, n_classes)
    all_f1s_matrix = np.array(all_f1s).reshape(args.n_folds, n_classes)

    # Calcula el promedio y desviación estándar por fold
    mean_precisions_per_fold = np.mean(all_precisions_matrix, axis=1)
    std_precisions_per_fold = np.std(all_precisions_matrix, axis=1)

    mean_recalls_per_fold = np.mean(all_recalls_matrix, axis=1)
    std_recalls_per_fold = np.std(all_recalls_matrix, axis=1)

    mean_specificities_per_fold = np.mean(all_specificities_matrix, axis=1)
    std_specificities_per_fold = np.std(all_specificities_matrix, axis=1)

    mean_accs_per_fold = np.mean(all_accs_matrix, axis=1)
    std_accs_per_fold = np.std(all_accs_matrix, axis=1)

    mean_f1s_per_fold = np.mean(all_f1s_matrix, axis=1)
    std_f1s_per_fold = np.std(all_f1s_matrix, axis=1)

    # calcula el promedio y desviación estándar de estos valores
    mean_precision = np.mean(mean_precisions_per_fold)
    std_precision = np.mean(std_precisions_per_fold)

    mean_recall = np.mean(mean_recalls_per_fold)
    std_recall = np.mean(std_recalls_per_fold)

    mean_specificity = np.mean(mean_specificities_per_fold)
    std_specificity = np.mean(std_specificities_per_fold)

    mean_acc = np.mean(mean_accs_per_fold)
    std_acc = np.mean(std_accs_per_fold)

    mean_f1 = np.mean(mean_f1s_per_fold)
    std_f1 = np.mean(std_f1s_per_fold)

    os.makedirs(f"{dir_path}average/{model_name}/", exist_ok=True)
    with open(f"{dir_path}average/{model_name}/metrics_summary.txt", 'w') as f:
        
        # Metrics per Fold and per Class
        for fold_idx, metrics in enumerate(all_metrics_per_fold):
            f.write(f"Metrics for Fold {fold_idx + 1}:\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            
            # Metrics averaged over classes for this fold
            f.write("\nAverage metrics for classes in this fold:\n")
            f.write(f"Precision: {mean_precisions_per_fold[fold_idx]:.4f} ± {std_precisions_per_fold[fold_idx]:.4f}\n")
            f.write(f"Recall: {mean_recalls_per_fold[fold_idx]:.4f} ± {std_recalls_per_fold[fold_idx]:.4f}\n")
            f.write(f"Specificity: {mean_specificities_per_fold[fold_idx]:.4f} ± {std_specificities_per_fold[fold_idx]:.4f}\n")
            f.write(f"Accuracy: {mean_accs_per_fold[fold_idx]:.4f} ± {std_accs_per_fold[fold_idx]:.4f}\n")
            f.write(f"F1-Score: {mean_f1s_per_fold[fold_idx]:.4f} ± {std_f1s_per_fold[fold_idx]:.4f}\n")
            f.write("\n-----------------------------\n")
        
        # Global Metrics averaged over all Folds and all Classes
        f.write("\nOverall Average Metrics (averaged over all folds and classes):\n")
        f.write(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}\n")
        f.write(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}\n")
        f.write(f"Mean Precision: {mean_precision:.4f} ± {std_precision:.4f}\n")
        f.write(f"Mean Recall: {mean_recall:.4f} ± {std_recall:.4f}\n")
        f.write(f"Mean Specificity: {mean_specificity:.4f} ± {std_specificity:.4f}\n")


if __name__ == '__main__':
    main()




