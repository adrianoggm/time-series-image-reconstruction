#!/home/adriano/Escritorio/TFG/venv/bin/python3

import argparse
from datasets import load_dataset

from datasets import load_dataset, Value
from datasets.features import Features, ClassLabel
from PIL import Image
import torch.nn.functional as F
import os
from tqdm.notebook import tqdm
import torch
import numpy as np

from huggingface_hub import login

# !pip install huggingface_hub &>> instal.log
# !huggingface-cli login 
# print("Insert huggingface token:")
# !huggingface-cli login
# from huggingface_hub import notebook_login
# notebook_login()

""" $ nohup ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loto --huggingface-token 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  > upload_rp_loto.log & """

def img_to_tensor(im):
  return torch.tensor(np.array(im.convert('RGB'))/255).permute(2, 0, 1).unsqueeze(0) * 2 - 1

def tensor_to_image(t):
  return Image.fromarray(np.array(((t.squeeze().permute(1, 2, 0)+1)/2).clip(0, 1)*255).astype(np.uint8))

def gather(consts: torch.Tensor, t: torch.Tensor):
    """Gather consts for $t$ and reshape to feature map shape"""
    c = consts.gather(-1, t)
    return c.reshape(-1, 1, 1, 1)

def upload_data(dataset_name="WISDM", dataset_folder="/home/adriano/Escritorio/TFG/data/WISDM/", sampling="loso", generateKFOLD=True, FOLDS_N=3, huggingfaceToken=""):
    DRIVE_RP_FOLDER = f"{dataset_folder}plots/recurrence_plot/sampling_{sampling}"
    HUGG_DATASET_NAME = f"Adrianoggm/{dataset_name}-mod-recurrence-plot-sampling-{sampling}"
    HUGG_DATASET_IS_PRIVATE = True
    print(DRIVE_RP_FOLDER,HUGG_DATASET_NAME,HUGG_DATASET_NAME)
    login(huggingfaceToken)

    # val_dataset = load_dataset("imagefolder", data_dir=DRIVE_RP_FOLDER, split="validation")
    if not generateKFOLD:
        print("Uploading Train dataset")
        train_dataset = load_dataset("imagefolder", data_dir=DRIVE_RP_FOLDER, split="train")
        train_dataset.push_to_hub(HUGG_DATASET_NAME, private=HUGG_DATASET_IS_PRIVATE)
        #upload Validation dataset
        print("Uploading Validation dataset")
        # val_dataset.push_to_hub(HUGG_DATASET_NAME, private=True)
    else:
        print("Uploading k-fold dataset")
        for fold in tqdm(range(FOLDS_N), "Uploading Fold"):
            print("fold:", fold)
            #print(f"{DRIVE_RP_FOLDER}/{FOLDS_N}-fold/fold-{fold}")
            train_dataset = load_dataset("imagefolder",data_dir=f"{DRIVE_RP_FOLDER}/{FOLDS_N}-fold/fold-{fold}", split=f"train")
            train_dataset.push_to_hub(f"{HUGG_DATASET_NAME}_fold_{fold}", private=HUGG_DATASET_IS_PRIVATE)
            train_dataset = load_dataset("imagefolder", data_dir=f"{DRIVE_RP_FOLDER}/{FOLDS_N}-fold/fold-{fold}", split=f"test")
            train_dataset.push_to_hub(f"{HUGG_DATASET_NAME}_fold_{fold}", private=HUGG_DATASET_IS_PRIVATE)

            # #upload Validation dataset
            # print("Uploading Validation dataset")
            # val_dataset.push_to_hub(f"{HUGG_DATASET_NAME}_fold_{fold}", private=True)






def main():
    '''
    Run after you create the recurrence plots with:
    $ ./generate_recurrence_plots.py
    or another:
    $ ./generate_recurrence_plots.py --data-name WISDM --n-folds 3 --data-folder /home/adriano/Escritorio/TFG/data/WISDM

    Then...
    
    $ chmod +x ./tgen/upload_recurrence_plots_to_huggingface.py 

    Examples of runs:
    $ nohup ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loso --huggingface-token YOUR_TOKEN > upload_rp.log &
    $ nohup ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loso --huggingface-token YOUR_TOKEN --data-name WISDM --n-folds 3 --data-folder /home/fmgarmor/proyectos/TGEN-timeseries-generation/data/WISDM/ > upload_rp.log &

    $ nohup ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loso --huggingface-token YOUR_TOKEN > upload_rp_loso.log &


    $ nohup ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loto --huggingface-token YOUR_TOKEN > upload_rp_loto.log &


    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--sampling', type=str, default="loso", help='the sampling technique')
    p.add_argument('--data-name', type=str, default="WISDM", help='the database name')
    p.add_argument('--data-folder', type=str, default="/home/adriano/Escritorio/TFG/data/WISDM/", help='the data folder path')
    p.add_argument('--n-folds', type=int, default=3, help='the number of k-folds')
    p.add_argument('--huggingface-token', required=True, type=str, help='the token')

    args = p.parse_args()

    data_folder = args.data_folder
    data_name = args.data_name
    FOLDS_N = args.n_folds
    huggingface_token = args.huggingface_token
    upload_data(data_name, data_folder, args.sampling, generateKFOLD=FOLDS_N>=0, FOLDS_N=FOLDS_N, huggingfaceToken=huggingface_token)

if __name__ == '__main__':
    main()