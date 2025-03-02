#!/home/fmgarmor/miot_env/bin/python3
import os
import argparse
import json
import accelerate
import torch
from tqdm import trange, tqdm
import pandas as pd

import k_diffusion as K
from tgen.utils import set_gpu

""" ./tgen/upload_recurrence_plots_to_huggingface.py --sampling loto --huggingface-token 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 > upload_rp_loto.log &

 """
def main():
    '''Examples of runs:
    $  nohup ./sample.py --config configs/config_wisdm_128x128_loto.json -n 2000  > sample-loto.log &

    $  nohup ./sample.py --config configs/config_wisdm_128x128_loso.json -n 2000  > sample-loso.log &


    $ we can sample for a specific model epoch (this will read the CSV f"{args.result_folder}{sampling_method}/model_steps_selection.csv"): 
    epoch,fold,class,step
    1000,0,0,68068
    1000,0,1,66066
    1000,0,2,66066
    1000,0,3,60000
    1000,0,4,68068
    1000,1,0,67067
    1000,1,1,65065
    1000,1,2,66066
    1000,1,3,67067
    1000,1,4,69069
    1000,2,0,67067
    1000,2,1,67067
    1000,2,2,68068
    1000,2,3,69069
    1000,2,4,60000
    2000,0,0, XXx

    --opt-model-epoch 1000"
    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')

    p.add_argument('--n-folds', type=int, default=3, help='the number of folds')
    p.add_argument('--n-classes', type=int, default=5, help='the number of classes')

    p.add_argument('--batch-size', type=int, default=16,
                   help='the batch size')
    p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",
                   help='the model config')
    p.add_argument('-n', type=int, default=1000,
                   help='the number of images to sample')
    
    p.add_argument('--result-folder', type=str, default='results/',
                   help='the output folder')
    p.add_argument('--steps', type=int, default=50, help='the number of denoising steps')
    p.add_argument('--best-model', action="store_true", help='if set, use the best model; else, use the last model')
    p.add_argument('--opt-model-epoch', type=str, default='', help='this will ignore --best-model load; here, we can sample for a specific model epoch')

    args = p.parse_args()

    set_gpu(args.gpu_id)


    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    sampling_method = dataset_config["sampling"]
    use_best_model = args.best_model

    # getting optional model check points for every fold and class with singular step:
    # we can sample for a specific fold,class=step;fold,class=step; ... :
    # --opt-model-step "0,0=00001234;0,1=00005678;...
    opt_model_steps = {}
    if args.opt_model_epoch:
        df_steps = pd.read_csv(f"{args.result_folder}{sampling_method}/model_steps_selection.csv")
        print(df_steps)
        df_steps = df_steps[df_steps['epoch'].astype(int) == int(args.opt_model_epoch)]
        df_steps.dropna(subset=['fold', 'class', 'step'], inplace=True)
        print(df_steps)
        opt_model_steps = {(int(row['fold']), int(row['class'])): int(row['step']) for _, row in df_steps.iterrows()}
        print(opt_model_steps)


    # TODO: allow non-square input sizes
    assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
    size = model_config['input_size']

    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print('Using device:', device, flush=True)

    for cl in range(args.n_classes):
        print(f"Sampling [Class-{cl}]")
        for fold in range(args.n_folds):
            print(f"Sampling [Class-{cl}] fold-{fold}")
            chp_path = ""
            if use_best_model:
                print("Getting the best bodel checkpoint")
                metrics_path = f"{args.result_folder}{sampling_method}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_metrics.csv"
                met_df = pd.read_csv(metrics_path)
                df_sorted = met_df.sort_values(by='fid')
                steps_sorted_by_fid = df_sorted['step'].tolist()
                for step in steps_sorted_by_fid:
                    chp_path = f'{args.result_folder}{sampling_method}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_{step:08}.pth'
                    if os.path.exists(chp_path):
                        print(f"Found model check point for the step {step}")
                        break
                    else:
                        print(f"Not found model check point for the step {step}")
            elif args.opt_model_epoch == "":
                print("Getting the last model checkpoint")
                with open(f"{args.result_folder}{sampling_method}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_state.json", 'r') as fcc_file:
                    fcc_data = json.load(fcc_file)
                    print(fcc_data["latest_checkpoint"])
                    chp_path = fcc_data["latest_checkpoint"]
            else:
                if (fold, cl) in opt_model_steps:
                    step = opt_model_steps[(fold, cl)]
                    chp_path = f'{args.result_folder}{sampling_method}/fold_{fold}/class_{cl}/model_fold_{fold}_class_{cl}_{step:08}.pth'
                    if os.path.exists(chp_path):
                        print(f"Found model check point for the step {step}")
                    else:
                        print(f"Not found model check point for the step {step}")
                        
                
            print("Checkpoint path selected:", chp_path)
            ckpt = torch.load(chp_path, map_location='cpu')
            # inner_model = K.config.make_model(config).eval().requires_grad_(False).to(device)
            inner_model = K.models.ImageDenoiserModelV1(
                model_config['input_channels'], # input channels
                model_config['mapping_out'], # mapping out
                model_config['depths'], # depths
                model_config['channels'], # channels
                model_config['self_attn_depths']
                ).eval().requires_grad_(False).to(device)

            inner_model.load_state_dict(ckpt["model"])
            # print(inner_model)
            accelerator.print('Parameters:', K.utils.n_params(inner_model))
            model = K.Denoiser(inner_model, sigma_data=model_config['sigma_data'])

            sigma_min = model_config['sigma_min']
            sigma_max = model_config['sigma_max']

            @torch.no_grad()
            @K.utils.eval_mode(model)
            def run(cl, fold):
                if accelerator.is_local_main_process:
                    tqdm.write('Sampling...')
                sigmas = K.sampling.get_sigmas_karras(args.steps, sigma_min, sigma_max, rho=7., device=device)
                def sample_fn(n):
                    x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                    x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_local_main_process)
                    return x_0
                x_0 = K.evaluation.compute_features(accelerator, sample_fn, lambda x: x, args.n, args.batch_size)
                if accelerator.is_main_process:
                    print("Sampling and saving samples")
                    path = f'{args.result_folder}{sampling_method}/fold_{fold}/class_{cl}/sample/'
                    os.makedirs(path, exist_ok=True)
                    for i, out in enumerate(x_0):
                        filename = f"{path}sample_{i:05}.png"
                        K.utils.to_pil_image(out).save(filename)
            try:
                print("Running Sampling")
                run(cl, fold)
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()