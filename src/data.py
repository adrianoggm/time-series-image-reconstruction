#!/home/fmgarmor/miot_env/bin/python3

'''
Generation of images for train/test and assess the quality of synthetic.
Synthetic images are the train set.
TEST set are real data, which are used to validate the model in the training phase
Test set are real data, which is used to test the trained model
'''

from functools import partial
import shutil
from PIL import Image
import glob
import pandas as pd
import os
import torch
from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm.auto import trange, tqdm

from huggingface_hub import login
from datasets import load_dataset
import argparse

import k_diffusion as K

from utils import set_gpu


def main():
  '''Examples of runs:
  ** LOSO APPROACH
  $  nohup ./tgen/data.py --config configs/config_wisdm_128x128_loso.json --prefix exp-classes-all-classes --class-names 0,1,2,3,4 --splits 0,1,2 > data_splits.log &
  
  - only 2 clases (1, 3) and only the split 3 (real train images)
  $  nohup ./tgen/data.py --prefix exp-classes-1-3 --class-names 1,3 --splits 2 > data_splits.log &


  ** LOTO approach
  $  nohup ./tgen/data.py --config configs/config_wisdm_128x128_loto.json --prefix exp-classes-all-classes --class-names 0,1,2,3,4 --splits 0,1,2 > data_splits-loto.log &

  '''
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')

  p.add_argument('--batch-size', type=int, default=16,
                  help='the batch size')
  p.add_argument('--image-size', type=int, default=128,
                  help='the image resolution size')
  p.add_argument('--n-folds', type=int, default=3,
                  help='the number of folds')
  p.add_argument('--class-names', type=str, default="0,1,2,3,4", #numbers: [0:"Walking",1:"Jogging",2:"Stairs",3:"Sitting",4:"Standing"]
                  help='the number of classes')
  p.add_argument('--splits', type=str, default="0,1,2", #numbers: 0: train split (synthetic images); 1: test split, 2: real train (real images for train)
                  help='the number of classes')
  p.add_argument('--prefix', type=str, default='exp-all-classes',
                  help='the output prefix')
  p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",help='the configuration file')
  
  args = p.parse_args()

  set_gpu(args.gpu_id)

  # Copy TEST + synthetics images
  folds = args.n_folds
  classes = [int(c) for c in args.class_names.split(",")]
  splits = [int(c) for c in args.splits.split(",")]
  print("Classes")

  config = K.config.load_config(open(args.config))
  # model_config = config['model']
  dataset_config = config['dataset']
  sampling_method = dataset_config["sampling"]

  CLASS_NAMES = ["Walking","Jogging","Stairs","Sitting","Standing"]
  tf = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.image_size),
        K.augmentation.KarrasAugmentationPipeline(0.0),
    ])


  for fold in range(folds):
    x_train, y_train, subjects_train, file_id_train,  x_real_train, y_real_train, subjects_real_train, file_id_real_train,  x_test, y_test, subjects_test, file_id_test, x_val, y_val, subjects_val, file_id_val = [], [], [], 0, [], [], [], 0, [], [], [], 0, [], [], [], 0

    for cl_idx in classes:
      cl = CLASS_NAMES[cl_idx]
      print(f"[Fold-{fold}] Coping files class: {cl} [{cl_idx}]")
      # test_src = f"data/WISDM/plots/recurrence_plot/{folds}-fold/fold-{fold}/test/{cl_idx}/"
      # val_src = f"data/WISDM/plots/recurrence_plot/validation/{cl_idx}/"
      
      #----------------------------------------------------
      #SYNTHETIC (TRAIN) files are the train dataset
      #----------------------------------------------------
      if 0 in splits:
        print("Generating SYNTHETIC (TRAIN) split")
        synth_dir = f"results/{sampling_method}/fold_{fold}/class_{cl_idx}/sample/"
        synth_dst_dir = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/train/"
        os.makedirs(synth_dst_dir, exist_ok=True)

        for i,f in enumerate( os.listdir(synth_dir)):
          # print(f"From: {synth_dir}{f}. TO: {dst}x{next_f+i}.png")      
          shutil.copyfile(f"{synth_dir}{f}", f"{synth_dst_dir}synth_{file_id_train}.png")
          x_train.append(f"synth_{file_id_train}.png")
        #   subjects_train.append(-1) #synthetic will be subject ID -1
          y_train.append(cl) #here, the class label is the index of class list
          file_id_train += 1  
        pd.DataFrame({"filename": x_train, "label": y_train}).to_csv(f"{synth_dst_dir}/training_labels.csv")

      #----------------------------------------------------
      #TEST files
      #----------------------------------------------------
      if 1 in splits:
        print("Generating TEST split")
        test_dst = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/test/"
        os.makedirs(test_dst, exist_ok=True)
        

        test_set = load_dataset(f"{dataset_config['location']}{fold}", split="test")
        test_set = test_set.filter(lambda t: t["label"] == cl_idx)
        test_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key="image"))
        test_dl = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        for batch in tqdm(test_dl):
          reals, reals_no_aug, aug_cond = batch["image"]
          # print(reals_no_aug.shape)
          for img in reals_no_aug:
            # print(img.shape)
            K.utils.to_pil_image(img).save(f"{test_dst}test_{file_id_test}.png")
            x_test.append(f"test_{file_id_test}.png")
            y_test.append(cl)
            file_id_test += 1
        pd.DataFrame({"filename": x_test, "label": y_test}).to_csv(f"{test_dst}/test_labels.csv")

      
      #----------------------------------------------------
      #REAL TRAIN files
      #----------------------------------------------------
      if 2 in splits:
        print("Generating REAL TRAIN split")
        real_train_dst = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/real_train/"
        os.makedirs(real_train_dst, exist_ok=True)

        real_train_set = load_dataset(f"{dataset_config['location']}{fold}", split="train")
        real_train_set = real_train_set.filter(lambda t: t["label"] == cl_idx)
        real_train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key="image"))
        real_train_dl = data.DataLoader(real_train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        for batch in tqdm(real_train_dl):
          reals, reals_no_aug, aug_cond = batch["image"]
          # print(reals_no_aug.shape)
          for img in reals_no_aug:
            # print(img.shape)
            K.utils.to_pil_image(img).save(f"{real_train_dst}real_train_{file_id_real_train}.png")
            x_real_train.append(f"real_train_{file_id_real_train}.png")
            y_real_train.append(cl)
            file_id_real_train += 1
        pd.DataFrame({"filename": x_real_train, "label": y_real_train}).to_csv(f"{real_train_dst}/real_training_labels.csv")

      #----------------------------------------------------
      #VALIDATION files
      #----------------------------------------------------
      if 3 in splits:
        print("Generating VALIDATION split")
        val_dst = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/validation/"
        os.makedirs(val_dst, exist_ok=True)

        val_set = load_dataset(f"{dataset_config['location']}{fold}", split="validation")
        val_set = val_set.filter(lambda t: t["label"] == cl_idx)
        val_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key="image"))
        val_dl = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        for batch in tqdm(val_dl):
          reals, reals_no_aug, aug_cond = batch["image"]
          # print(reals_no_aug.shape)
          for img in reals_no_aug:
            # print(img.shape)
            K.utils.to_pil_image(img).save(f"{val_dst}val_{file_id_val}.png")
            x_val.append(f"val_{file_id_val}.png")
            y_val.append(cl)
            file_id_val += 1
        pd.DataFrame({"filename": x_val, "label": y_val}).to_csv(f"{val_dst}/val_labels.csv")


if __name__ == '__main__':
    main()