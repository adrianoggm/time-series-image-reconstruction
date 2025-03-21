#!/home/adriano/Escritorio/TFG/venv/bin/python3

import os
import argparse
import wandb

def main():
    p = argparse.ArgumentParser(description="Load models from a directory and upload to Wandb.")
    p.add_argument('--data-folder', type=str, default="results", help='The folder path containing the models')
    p.add_argument('--wandb-entity', type=str, default="frangam", help='the wandb entity name')
    p.add_argument('--wandb-group', type=str, help='the wandb group name')
    p.add_argument('--wandb-project', type=str, default="diffusion-ts-rp", help='the wandb project name (specify this to enable wandb)')
    args = p.parse_args()

    # Initialize Wandb
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group)

    # Directory where the models are stored
    destination_dir = os.path.join(args.data_folder, "selected_models")

    # Ensure the destination directory exists
    if not os.path.exists(destination_dir):
        print(f"Error: Directory {destination_dir} does not exist!")
        return

    # Iterate over each model in the directory and upload to Wandb
    for model_file in os.listdir(destination_dir):
        model_path = os.path.join(destination_dir, model_file)
        if model_file.endswith('.pth'):
            sampling = model_file.split("_")[2]
            epoch = model_file.split("epoch_")[-1].split("_")[0]
            fold = model_file.split("fold_")[-1].split("_")[0]
            theclass = model_file.split("class_")[-1].split("_")[0]
            
            print(f"Uploading model {model_path} to Wandb...")
            
            artifact = wandb.Artifact(f'{model_file.split(".")[0]}',
                                      type='model',
                                      metadata={
                                          'format': 'diffusion-model',
                                          'class': theclass,
                                          'fold': fold,
                                          'sampling': sampling
                                      })
            artifact.add_file(model_path)
            # wandb.log_artifact(artifact, aliases=[f"epoch-{epoch}"])
            wandb.log_artifact(artifact, aliases=[f"v0"])


    # Finish Wandb run
    wandb.finish()

if __name__ == '__main__':
    main()
