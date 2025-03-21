#!/home/fmgarmor/miot_env/bin/python3

import math
from functools import partial
import shutil
from PIL import Image
import glob
import pandas as pd
import os
import torch
from torchvision import datasets, transforms, utils as tv_utils
from torch.utils import data
from tqdm.auto import trange, tqdm

from datasets import load_dataset
import argparse

import seaborn as sns

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input

import numpy as np


sns.set_style("darkgrid")

import matplotlib.pyplot as plt

from torchvision import  utils



import k_diffusion as K

def plotImages(images_arr, path, fold, dltype="train"):
    os.makedirs(f"{path}{dltype}/", exist_ok=True)
    print(images_arr.shape)
    fig, axes = plt.subplots(math.ceil(images_arr.shape[0] ** 0.5), math.ceil(images_arr.shape[0] ** 0.5), figsize=(1,1))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        # transforms.functional.to_pil_image( torch.from_numpy(img.reshape((img.shape[2], img.shape[0], img.shape[1])))).save(f"{path}{dltype}/fold_{fold}_grid.png")

    plt.tight_layout()
    plt.savefig(f"{path}{dltype}/fold_{fold}_img_array.png", dpi=300, bbox_inches='tight')

    plt.show()

def plot(path, fold, dl, dltype="train", max_batches=1):
    os.makedirs(f"{path}{dltype}/", exist_ok=True)
    # for i,batch in enumerate(tqdm(dl)):
    #     batch = batch[0]
        # plotImages(batch, path, fold, dltype)
        # print(batch.shape)
        # batch = torch.from_numpy(batch.reshape((batch.shape[0], batch.shape[3], batch.shape[1], batch.shape[2])))
        # print(batch.shape)
        # batch = torch.from_numpy(batch)

    # Creamos un grid de segun el batchsize de imágenes
    imgs_array, labels = next(dl)
    # grid_imgs = make_grid(batch, nrow=4)
    
    imgs_array = np.transpose(imgs_array, (0, 3, 1, 2))
    imgs_tensor = torch.from_numpy(imgs_array)


    grid = tv_utils.make_grid(imgs_tensor, nrow=math.ceil(imgs_array.shape[0] ** 0.5), padding=2, pad_value=1)
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig(f"{path}{dltype}/fold_{fold}_grid.png", dpi=300, bbox_inches='tight')
    plt.show()
    # transforms.functional.to_pil_image(grid).save(f"{path}{dltype}/fold_{fold}_grid_{i}.png")
    # if i+1 == max_batches:
        # break 

def main():
    '''Examples of runs:

    $  nohup ./plots.py --sampling loso > plots.log &
    '''
    p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sampling', type=str, default="loso",
                    help='the sampling method')
    p.add_argument('--batch-size', type=int, default=64,
                    help='the batch size')
    p.add_argument('--image-size', type=int, default=128,
                    help='the image resolution size')
    p.add_argument('--n-folds', type=int, default=3,
                    help='the number of folds')
    p.add_argument('--prefix', type=str, default='exp-classes-all-classes',
                    help='the output prefix')
    p.add_argument('--plot-real-images', action="store_true") 
  
    args = p.parse_args()
    folds = args.n_folds
    img_size = args.image_size
    batch_size = args.batch_size

    if args.plot_real_images:
        train_idg = ImageDataGenerator(rescale=1./255,  validation_split=0.0)
        val_idg = ImageDataGenerator(rescale=1./255)
        test_idg = ImageDataGenerator(rescale=1./255)

        for fold in range(folds):
            train_path = f"results/evaluation_synthetic_quality/{args.sampling}/{args.prefix}/data/fold_{fold}/train/"
            test_path = f"results/evaluation_synthetic_quality/{args.sampling}/{args.prefix}/data/fold_{fold}/test/"
            val_path = f"results/evaluation_synthetic_quality/{args.sampling}/{args.prefix}/data/fold_{fold}/validation/"

            train_data = pd.read_csv(f'{train_path}training_labels.csv', dtype=str)[["filename", "label"]]
            test_data = pd.read_csv(f'{test_path}test_labels.csv', dtype=str)[["filename", "label"]]
            # val_data = pd.read_csv(f'{val_path}val_labels.csv', dtype=str)[["filename", "label"]]


            train_data_generator = train_idg.flow_from_dataframe(train_data, batch_size=batch_size, target_size=(img_size, img_size), directory = train_path,
                            x_col = "filename", y_col = "label",
                            # class_mode = "raw", 
                            class_mode="categorical",
                            shuffle = True, seed=33
                            )
            # valid_data_generator  = val_idg.flow_from_dataframe(val_data, batch_size=batch_size, target_size=(img_size, img_size), directory = val_path,
            #             x_col = "filename", y_col = "label",
            #             # class_mode = "raw", 
            #             class_mode="categorical",
            #             shuffle = True, seed=33
            #             )
            #to test the trained model, we used real test data unseen on training phase
            test_data_generator  = test_idg.flow_from_dataframe(test_data, batch_size=batch_size, target_size=(img_size, img_size), directory = test_path,
                        x_col = "filename", y_col = "label",
                        # class_mode = "raw", 
                        class_mode="categorical",
                        shuffle = False, #important shuffle=False to compare the prediction result
                        ) 
        
            
            # batch, labels = next(iter(train_data_generator))
            path = f"results/evaluation_synthetic_quality/{args.sampling}/{args.prefix}/plots/"
            plot(path, fold, train_data_generator, dltype="train", max_batches=1)
            # plot(path, fold, valid_data_generator, dltype="validation", max_batches=1)
            plot(path, fold, test_data_generator, dltype="test", max_batches=1)

    for cl in range(5):
        for fold in range(folds):
            train_dataset = load_dataset("imagefolder", data_dir=f"data/WISDM/plots/recurrence_plot/sampling_{args.sampling}/{folds}-fold/fold-{fold}/train/{cl}/")["train"]
            
            batch = []
            for i, image in enumerate(train_dataset):
                img = np.array(image["image"])
                print(img.shape)
                batch.append(img)
                if i==batch_size-1:
                    break
            print(np.array(batch).shape)
            grid = utils.make_grid(np.array(batch), nrow=math.ceil(batch_size ** 0.5), padding=0)
            K.utils.to_pil_image(grid).save(f"data/WISDM/plots/recurrence_plot/sampling_{args.sampling}/{folds}-fold/fold-{fold}/train_example_batches/{cl}/")

            # batch, labels = next(iter(train_data_generator))


if __name__ == '__main__':
    main()