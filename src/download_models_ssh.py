#!/home/fmgarmor/miot_env/bin/python3

import os
import pandas as pd
import argparse
import shutil

def copy_model_to_new_location(source_path, destination_path):
    shutil.copy(source_path, destination_path)

def main():
    '''
    Example of use:
    $ nohup ./tgen/download_models_ssh.py --sampling loso --data-folder results --csv-path results/loso/model_steps_selection.csv > logs/download_models-loso.log &

    $ nohup ./tgen/download_models_ssh.py --sampling loto --data-folder results --csv-path results/loto/model_steps_selection.csv > logs/download_models-loto.log &

    '''
    p = argparse.ArgumentParser(description="Copy selected models to a new location.")
    p.add_argument('--sampling', type=str, required=True, help='The sampling technique (e.g., "loso" or "loto")')
    p.add_argument('--data-folder', type=str, default="results", help='The folder path containing the models')
    p.add_argument('--csv-path', required=True, type=str, help='Path to the CSV file that specifies which models to copy')

    args = p.parse_args()
    df = pd.read_csv(args.csv_path)

    # Ensure the destination directory exists
    destination_dir = os.path.join(args.data_folder, "selected_models")
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
        print("creating folder:", destination_dir)
    print( df.iterrows())

    for index, row in df.iterrows():
        epoch = row['epoch']
        fold = row['fold']
        cl = row['class']
        step = row['step']
        
        source_path = f'{args.data_folder}/{args.sampling}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_{step:08}.pth'
        destination_path = os.path.join(destination_dir, f"model_sampling_{args.sampling}_epoch_{epoch}_fold_{fold}_class_{cl}_{step:08}.pth")
        
        print(f"Copying model from {source_path} to {destination_path}...")
        copy_model_to_new_location(source_path, destination_path)

if __name__ == '__main__':
    main()
