#!/home/adriano/Escritorio/TFG/venv/bin/python3

import os
import pandas as pd
import argparse
from huggingface_hub import Repository, HfApi
import torch


def upload_model_to_hf(chp_path, model_name, token):
    repo = Repository(local_dir=model_name, clone_from=f"frangam/{model_name}", use_auth_token=token)
    os.system(f"cp {chp_path} {model_name}/model.pth")
    repo.push_to_hub(commit_message="Adding model")


def main():
    p = argparse.ArgumentParser(description="Upload models to Hugging Face.")
    p.add_argument('--sampling', type=str, required=True, help='The sampling technique (e.g., "loso" or "loto")')
    p.add_argument('--data-folder', type=str, default="results", help='The folder path containing the models')
    p.add_argument('--huggingface-token', required=True, type=str, help='Your Hugging Face token')
    p.add_argument('--csv-path', required=True, type=str, help='Path to the CSV file that specifies which models to upload')

    args = p.parse_args()
    df = pd.read_csv(args.csv_path)

    for index, row in df.iterrows():
        epoch = row['epoch']
        fold = row['fold']
        cl = row['class']
        step = row['step']
        
        chp_path = f'{args.data_folder}/{args.sampling}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_{step:08}.pth'
        model_name = f"Diff-TSD_epoch_{epoch}_sampling_{args.sampling}_fold_{fold}_class_{cl}"
        
        print(f"Uploading model from {chp_path} to Hugging Face as {model_name}...")
        upload_model_to_hf(chp_path, model_name, args.huggingface_token)

if __name__ == '__main__':
    main()
