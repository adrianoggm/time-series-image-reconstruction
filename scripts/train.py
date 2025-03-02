#!/home/adriano/Escritorio/TFG/venv/bin/python3

import os
import argparse
from copy import deepcopy
from functools import partial
import math
import json
from pathlib import Path

import accelerate
import torch
from torch import nn, optim
from torch import multiprocessing as mp
from torch.utils import data
from torchvision import datasets, transforms, utils
from datasets import load_dataset
from tqdm.auto import trange, tqdm
import wandb
import k_diffusion as K

import tgen.utils as tu

def main():
    '''Examples of runs:

    $ nohup ./train.py --config configs/config_wisdm_128x128_loso.json --max-epochs 1000 > train_loso.log &
    
    $ nohup ./train.py --config configs/config_wisdm_128x128_loto.json --max-epochs 1000 > train_loto.log &


    No supported multi-gpu with accelerator now
    only one gpu 
    $ accelerate config, then:
    $ nohup accelerate launch ./train.py --max-epochs 2 > train.log &


    '''
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # p.add_argument('--device', type=int, default=2,
    #                help='the device; -1: if none')
    p.add_argument('--batch-size', type=int, default=16,
                   help='the batch size')
    p.add_argument('--max-epochs', type=int, default=1000,
                   help='the maximum epochs')
    p.add_argument('--class-names', type=str, default="0,1,2,3,4",
                   help='the class labels')
    p.add_argument('--folds', type=str, default="0,1,2", help='the specific k-folds')
    p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",
                   help='the configuration file')
    p.add_argument('--demo-every', type=int, default=500,
                   help='save a demo grid every this many steps')
    p.add_argument('--evaluate-every', type=int, default=5000,
                   help='save a demo grid every this many steps')
    p.add_argument('--evaluate-n', type=int, default=1000,
                   help='the number of samples to draw to evaluate')
    p.add_argument('--gns', action='store_true',
                   help='measure the gradient noise scale (DDP only)')
    p.add_argument('--grad-accum-steps', type=int, default=1,
                   help='the number of gradient accumulation steps')
    p.add_argument('--grow', type=str,
                   help='the checkpoint to grow from')
    p.add_argument('--grow-config', type=str,
                   help='the configuration file of the model to grow from')
    p.add_argument('--lr', type=float,
                   help='the learning rate')
    p.add_argument('--name', type=str, default='model',
                   help='the name of the run')
    p.add_argument('--num-workers', type=int, default=8,
                   help='the number of data loader workers')
    p.add_argument('--resume', type=str,
                   help='the checkpoint to resume from')
    p.add_argument('--sample-n', type=int, default=64,
                   help='the number of images to sample for demo grids')
    p.add_argument('--save-every', type=int, default=10000,
                   help='save every this many steps')
    p.add_argument('--seed', type=int,
                   help='the random seed')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--wandb-entity', type=str, default="frangam",
                   help='the wandb entity name')
    p.add_argument('--wandb-group', type=str,
                   help='the wandb group name')
    p.add_argument('--wandb-project', type=str, default="diffusion-ts-rp",
                   help='the wandb project name (specify this to enable wandb)')
    p.add_argument('--wandb-save-model', action='store_true',
                   help='save model to wandb')
    args = p.parse_args()

    #-- TODO delete this dict after experiments...
    run_ids={
        "loto-fold-2-cl-4-128x128": "run-20230514_201508-whsjamux",
        "loto-fold-1-cl-4-128x128": "run-20230514_145053-pbfywjox",
        "loto-fold-0-cl-4-128x128": "run-20230514_092657-xzo4miw4",
        
        "loto-fold-2-cl-3-128x128": "run-20230514_040556-kvcqvwso",
        "loto-fold-1-cl-3-128x128": "run-20230513_224508-fhmdk1e3",
        "loto-fold-0-cl-3-128x128": "run-20230513_144852-392ng5jt",

        "loto-fold-2-cl-2-128x128": "run-20230513_093033-bzhn0yb8",
        "loto-fold-1-cl-2-128x128": "run-20230513_041242-i0vnva2v",
        "loto-fold-0-cl-2-128x128": "run-20230512_225436-6yvmn1rz",

        "loto-fold-2-cl-1-128x128": "run-20230512_170152-qig09rle",
        "loto-fold-1-cl-1-128x128": "run-20230512_102438-ywy7or5j",
        "loto-fold-0-cl-1-128x128": "run-20230512_044828-aqpg4zw9",

        "loto-fold-2-cl-0-128x128": "run-20230511_233003-s7gkpke7",
        "loto-fold-1-cl-0-128x128": "run-20230511_170516-j1end0xv",
        "loto-fold-0-cl-0-128x128": "run-20230511_110519-7ysc8d7a",


        "loso-fold-2-cl-4-128x128": "run-20230508_210108-t4oczxum",
        "loso-fold-1-cl-4-128x128": "run-20230508_153222-ue8ifi80",
        "loso-fold-0-cl-4-128x128": "run-20230508_100621-9qymlve4",
        
        "loso-fold-2-cl-3-128x128": "run-20230508_043752-hkgz11uq",
        "loso-fold-1-cl-3-128x128": "run-20230507_231618-tdne2jji",
        "loso-fold-0-cl-3-128x128": "run-20230507_173756-fjneof0t",

        "loso-fold-2-cl-2-128x128": "run-20230507_121301-yyjdoa0j",
        "loso-fold-1-cl-2-128x128": "run-20230507_065516-zgqxcoma",
        "loso-fold-0-cl-2-128x128": "run-20230507_013737-cvtfijfq",

        "loso-fold-2-cl-1-128x128": "run-20230506_201610-wi6r8msg",
        "loso-fold-1-cl-1-128x128": "run-20230506_150109-357kcdkq",
        "loso-fold-0-cl-1-128x128": "run-20230506_094206-9lj172xw",

        "loso-fold-2-cl-0-128x128": "run-20230506_041525-q4ksq5qg",
        "loso-fold-1-cl-0-128x128": "run-20230505_224841-kulaz209",
        "loso-fold-0-cl-0-128x128": "run-20230505_171823-2y7wbtbd"
    }
    #---


    mp.set_start_method(args.start_method)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = K.config.load_config(open(args.config))
    model_config = config['model']
    dataset_config = config['dataset']
    sampling_method = dataset_config["sampling"]
    opt_config = config['optimizer']
    sched_config = config['lr_sched']
    ema_sched_config = config['ema_sched']
    classes = [int(c) for c in args.class_names.split(",")]
    specific_folds = [int(f) for f in args.folds.split(",")]
    print("Classes to train:", classes)
    
    for cl_idx in tqdm(range(len(classes)), desc="Class"):
        class_label = classes[cl_idx]
        print("Processing class:", class_label)
        for current_fold_i in tqdm(range(len(specific_folds)), desc=f"[Class {class_label}] Fold"):
            current_fold = specific_folds[current_fold_i]
            os.makedirs(f"demo/{sampling_method}/fold_{current_fold}/class_{class_label}", exist_ok=True)
            os.makedirs(f"results/{sampling_method}/fold_{current_fold}/class_{class_label}", exist_ok=True)

            # TODO: allow non-square input sizes
            assert len(model_config['input_size']) == 2 and model_config['input_size'][0] == model_config['input_size'][1]
            size = model_config['input_size']

            ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=model_config['skip_stages'] > 0)
            accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs], gradient_accumulation_steps=args.grad_accum_steps)
            device = accelerator.device #if args.device < 0 else args.device
            print(f'Process {accelerator.process_index} using device: {device}', flush=True)

            if args.seed is not None:
                seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes], generator=torch.Generator().manual_seed(args.seed))
                torch.manual_seed(seeds[accelerator.process_index])

            # inner_model = K.config.make_model(config)
            inner_model = K.models.ImageDenoiserModelV1(
                model_config['input_channels'], # input channels
                model_config['mapping_out'], # mapping out
                model_config['depths'], # depths
                model_config['channels'], # channels
                model_config['self_attn_depths'],
                patch_size=model_config["patch_size"]
                ) # self attention


            inner_model_ema = deepcopy(inner_model)
            if accelerator.is_main_process:
                print('Parameters:', K.utils.n_params(inner_model))
            
            state_path = Path(f'results/{sampling_method}/fold_{current_fold}/class_{class_label}/{args.name}_fold_{current_fold}_class_{class_label}_state.json')

            # If logging to wandb, initialize the run
            use_wandb = accelerator.is_main_process and args.wandb_project
            if use_wandb:
                log_config = vars(args)
                log_config['config'] = config
                log_config['parameters'] = K.utils.n_params(inner_model)
                run_id = f"{sampling_method}-fold-{current_fold}-cl-{class_label}-{size[0]}x{size[1]}"
                run_id = run_ids[run_id].split("-")[-1] #TODO keep this line only for this experiment (because, run ids did not be generated manually the firs time !!)
                if state_path.exists() or args.resume:
                    wandb.init(id=run_id, resume="must", project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, save_code=True, tags=[f"Fold-{current_fold}", f"Class-{class_label}", f"{model_config['input_size'][0]}x{model_config['input_size'][1]}", f"Sampling-{sampling_method}"])
                else:
                    wandb.init(id=run_id, project=args.wandb_project, entity=args.wandb_entity, group=args.wandb_group, config=log_config, save_code=True, tags=[f"Fold-{current_fold}", f"Class-{class_label}", f"{model_config['input_size'][0]}x{model_config['input_size'][1]}", f"Sampling-{sampling_method}"])
                print("inited wandb")
            if opt_config['type'] == 'adamw':
                opt = optim.AdamW(inner_model.parameters(),
                                lr=opt_config['lr'] if args.lr is None else args.lr,
                                betas=tuple(opt_config['betas']),
                                eps=opt_config['eps'],
                                weight_decay=opt_config['weight_decay'])
            elif opt_config['type'] == 'sgd':
                opt = optim.SGD(inner_model.parameters(),
                                lr=opt_config['lr'] if args.lr is None else args.lr,
                                momentum=opt_config.get('momentum', 0.),
                                nesterov=opt_config.get('nesterov', False),
                                weight_decay=opt_config.get('weight_decay', 0.))
            else:
                raise ValueError('Invalid optimizer type')

            if sched_config['type'] == 'inverse':
                sched = K.utils.InverseLR(opt,
                                        inv_gamma=sched_config['inv_gamma'],
                                        power=sched_config['power'],
                                        warmup=sched_config['warmup'])
            elif sched_config['type'] == 'exponential':
                sched = K.utils.ExponentialLR(opt,
                                            num_steps=sched_config['num_steps'],
                                            decay=sched_config['decay'],
                                            warmup=sched_config['warmup'])
            elif sched_config['type'] == 'constant':
                sched = optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
            else:
                raise ValueError('Invalid schedule type')

            assert ema_sched_config['type'] == 'inverse'
            ema_sched = K.utils.EMAWarmup(power=ema_sched_config['power'],
                                        max_value=ema_sched_config['max_value'])

            tf = transforms.Compose([
                transforms.Resize(size[0], interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(size[0]),
                K.augmentation.KarrasAugmentationPipeline(model_config['augment_prob']),
                
            ])

            if dataset_config['type'] == 'imagefolder':
                train_set = K.utils.FolderOfImages(dataset_config['location'], transform=tf)
            elif dataset_config['type'] == 'cifar10':
                train_set = datasets.CIFAR10(dataset_config['location'], train=True, download=True, transform=tf)
            elif dataset_config['type'] == 'mnist':
                train_set = datasets.MNIST(dataset_config['location'], train=True, download=True, transform=tf)
            elif dataset_config['type'] == 'huggingface':
                train_set = load_dataset(f"{dataset_config['location']}{current_fold}", split="train")
                train_set = train_set.filter(lambda t: t["label"] == class_label)
                train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key=dataset_config['image_key']))
                #train_set = train_set['train']
                print("FOLD:", current_fold, "CLASS LABEL:", class_label, "DATA LENGTH:", len(train_set))
            else:
                raise ValueError('Invalid dataset type')

            if accelerator.is_main_process:
                try:
                    print('Number of items in dataset:', len(train_set))
                except TypeError:
                    pass

            image_key = dataset_config.get('image_key', 0)
            print("IMAGE KEY:", image_key)

            train_dl = data.DataLoader(train_set, args.batch_size, shuffle=True, drop_last=True,
                                    num_workers=args.num_workers, persistent_workers=True)

            if args.grow:
                if not args.grow_config:
                    raise ValueError('--grow requires --grow-config')
                ckpt = torch.load(args.grow, map_location='cpu')
                old_config = K.config.load_config(open(args.grow_config))
                old_inner_model = K.config.make_model(old_config)
                old_inner_model.load_state_dict(ckpt['model_ema'])
                if old_config['model']['skip_stages'] != model_config['skip_stages']:
                    old_inner_model.set_skip_stages(model_config['skip_stages'])
                if old_config['model']['patch_size'] != model_config['patch_size']:
                    old_inner_model.set_patch_size(model_config['patch_size'])
                inner_model.load_state_dict(old_inner_model.state_dict())
                del ckpt, old_inner_model

            inner_model, inner_model_ema, opt, train_dl = accelerator.prepare(inner_model, inner_model_ema, opt, train_dl)
            if use_wandb:
                wandb.watch(inner_model)
            if args.gns:
                gns_stats_hook = K.gns.DDPGradientStatsHook(inner_model)
                gns_stats = K.gns.GradientNoiseScale()
            else:
                gns_stats = None
            sigma_min = model_config['sigma_min']
            sigma_max = model_config['sigma_max']
            sample_density = K.config.make_sample_density(model_config)

            model = K.layers.Denoiser(inner_model, sigma_data=model_config["sigma_data"]).to(device) #K.config.make_denoiser_wrapper(config)(inner_model)
            model_ema = K.layers.Denoiser(inner_model_ema, sigma_data=model_config["sigma_data"]).to(device) #K.config.make_denoiser_wrapper(config)(inner_model_ema)

            # state_path = Path(f'results/{sampling_method}/fold_{current_fold}/class_{class_label}/{args.name}_fold_{current_fold}_class_{class_label}_state.json')

            if state_path.exists() or args.resume:
                if args.resume:
                    ckpt_path = args.resume
                if not args.resume:
                    state = json.load(open(state_path))
                    ckpt_path = state['latest_checkpoint']
                if accelerator.is_main_process:
                    print(f'Resuming from {ckpt_path}...')
                
                ckpt = torch.load(ckpt_path, map_location='cpu')
                accelerator.unwrap_model(model.inner_model).load_state_dict(ckpt['model'])
                accelerator.unwrap_model(model_ema.inner_model).load_state_dict(ckpt['model_ema'])
                opt.load_state_dict(ckpt['opt'])
                sched.load_state_dict(ckpt['sched'])
                ema_sched.load_state_dict(ckpt['ema_sched'])
                epoch = ckpt['epoch'] + 1
                step = ckpt['step'] + 1
                if args.gns and ckpt.get('gns_stats', None) is not None:
                    gns_stats.load_state_dict(ckpt['gns_stats'])
                print(f'Resuming from {ckpt_path}... Epoch {epoch} - Step {step}')

                del ckpt
            else:
                epoch = 0
                step = 0

            evaluate_enabled = args.evaluate_every > 0 and args.evaluate_n > 0
            if evaluate_enabled:
                extractor = K.evaluation.InceptionV3FeatureExtractor(device=device)
                train_iter = iter(train_dl)
                if accelerator.is_main_process:
                    print('Computing features for reals...')
                reals_features = K.evaluation.compute_features(accelerator, lambda x: next(train_iter)[image_key][1], extractor, args.evaluate_n, args.batch_size)
                if accelerator.is_main_process:
                    metrics_log = K.utils.CSVLogger(f'results/{sampling_method}/fold_{current_fold}/class_{class_label}/{args.name}_fold_{current_fold}_class_{class_label}_metrics.csv', ['step', 'fid', 'kid'])
                del train_iter

            @torch.no_grad()
            @K.utils.eval_mode(model)
            def demo():
                if accelerator.is_main_process:
                    tqdm.write('Sampling...')
                filename = f'demo/{sampling_method}/fold_{current_fold}/class_{class_label}/{args.name}_fold_{current_fold}_class_{class_label}_demo_{step:08}.png'
                n_per_proc = math.ceil(args.sample_n / accelerator.num_processes)
                x = torch.randn([n_per_proc, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
                x_0 = K.sampling.sample_lms(model, x, sigmas, disable=not accelerator.is_main_process)
                x_0 = accelerator.gather(x_0)[:args.sample_n]
                if accelerator.is_main_process:
                    grid = utils.make_grid(x_0, nrow=math.ceil(args.sample_n ** 0.5), padding=0)
                    K.utils.to_pil_image(grid).save(filename)
                    if use_wandb:
                        wandb.log({'demo_grid': wandb.Image(filename)}, step=step)

            @torch.no_grad()
            @K.utils.eval_mode(model)
            def evaluate():
                if not evaluate_enabled:
                    return
                if accelerator.is_main_process:
                    tqdm.write('Evaluating...')
                sigmas = K.sampling.get_sigmas_karras(50, sigma_min, sigma_max, rho=7., device=device)
                def sample_fn(n):
                    x = torch.randn([n, model_config['input_channels'], size[0], size[1]], device=device) * sigma_max
                    x_0 = K.sampling.sample_lms(model, x, sigmas, disable=True)
                    return x_0
                fakes_features = K.evaluation.compute_features(accelerator, sample_fn, extractor, args.evaluate_n, args.batch_size)
                if accelerator.is_main_process:
                    fid = K.evaluation.fid(fakes_features, reals_features)
                    kid = K.evaluation.kid(fakes_features, reals_features)
                    print(f'FID: {fid.item():g}, KID: {kid.item():g}')
                    if accelerator.is_main_process:
                        metrics_log.write(step, fid.item(), kid.item())
                    if use_wandb:
                        wandb.log({'FID': fid.item(), 'KID': kid.item()}, step=step)

            def save():
                accelerator.wait_for_everyone()
                filename = f'results/{sampling_method}/fold_{current_fold}/class_{class_label}/{args.name}_fold_{current_fold}_class_{class_label}_{step:08}.pth'
                if accelerator.is_main_process:
                    tqdm.write(f'Saving to {filename}...')
                obj = {
                    'model': accelerator.unwrap_model(model.inner_model).state_dict(),
                    'model_ema': accelerator.unwrap_model(model_ema.inner_model).state_dict(),
                    'opt': opt.state_dict(),
                    'sched': sched.state_dict(),
                    'ema_sched': ema_sched.state_dict(),
                    'epoch': epoch,
                    'step': step,
                    'gns_stats': gns_stats.state_dict() if gns_stats is not None else None,
                }
                accelerator.save(obj, filename)
                if accelerator.is_main_process:
                    state_obj = {'latest_checkpoint': filename}
                    json.dump(state_obj, open(state_path, 'w'))
                if args.wandb_save_model and use_wandb:
                    wandb.save(filename)

            try:
                any_training = False
                while epoch <= args.max_epochs:
                    for batch in tqdm(train_dl, disable=not accelerator.is_main_process, desc=f"[Class {class_label} - Fold {current_fold}] Epoch {epoch}"):
                        any_training = True
                        with accelerator.accumulate(model):
                            
                            reals, reals_no_aug, aug_cond = batch[image_key]

                            # grid = utils.make_grid(reals, nrow=math.ceil(len(reals) ** 0.5), padding=0)
                            # K.utils.to_pil_image(grid).save(f'results/{sampling_method}/fold_{current_fold}/{args.name}_reals_{step:08}.png')
                            # grid = utils.make_grid(reals_no_aug, nrow=math.ceil(len(reals_no_aug) ** 0.5), padding=0)
                            # K.utils.to_pil_image(grid).save(f'results/{sampling_method}/fold_{current_fold}/{args.name}_reals-noaug_{step:08}.png')

                            noise = torch.randn_like(reals_no_aug)
                            # sigma = sample_density([reals_no_aug.shape[0]], device=device)
                            sigma = torch.distributions.LogNormal(model_config["sigma_sample_density"]["mean"], model_config["sigma_sample_density"]["std"]).sample([reals_no_aug.shape[0]]).to(device)

                            losses = model.loss(reals_no_aug, noise, sigma) #, aug_cond=aug_cond)
                            losses_all = accelerator.gather(losses)
                            loss = losses_all.mean()
                            # print("LOSSES_ALL:", losses_all, "MEAN:",loss)
                            accelerator.backward(losses.mean())
                            if args.gns:
                                sq_norm_small_batch, sq_norm_large_batch = gns_stats_hook.get_stats()
                                gns_stats.update(sq_norm_small_batch, sq_norm_large_batch, reals_no_aug.shape[0], reals_no_aug.shape[0] * accelerator.num_processes)
                            opt.step()
                            sched.step() #learning scheduler just to kind of improve training stability
                            opt.zero_grad()
                            # if accelerator.sync_gradients:
                            #     ema_decay = ema_sched.get_value()
                            #     K.utils.ema_update(model, model_ema, ema_decay)
                            #     ema_sched.step()

                        if accelerator.is_main_process:
                            if step % 25 == 0:
                                if args.gns:
                                    tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}, gns: {gns_stats.get_gns():g}')
                                else:
                                    tqdm.write(f'Epoch: {epoch}, step: {step}, loss: {loss.item():g}')

                        if use_wandb:
                            log_dict = {
                                'epoch': epoch,
                                'loss': loss.item(),
                                'lr': sched.get_last_lr()[0],
                                # 'ema_decay': ema_decay,
                            }
                            if args.gns:
                                log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                            wandb.log(log_dict, step=step)

                        if step % args.demo_every == 0:
                            demo()

                        if evaluate_enabled and step > 0 and step % args.evaluate_every == 0:
                            evaluate()
                            save()

                        # if step > 0 and step % args.save_every == 0:
                        #     save()

                        step += 1
                    epoch += 1
                    
                if any_training:
                    if evaluate_enabled:
                        evaluate()
                    save()#save the last

                    if use_wandb:
                        log_dict = {
                            'epoch': epoch,
                            'loss': loss.item(),
                            'lr': sched.get_last_lr()[0],
                            # 'ema_decay': ema_decay,
                        }
                        if args.gns:
                            log_dict['gradient_noise_scale'] = gns_stats.get_gns()
                        wandb.log(log_dict, step=step)
                        wandb.finish()
            except KeyboardInterrupt:
                pass


if __name__ == '__main__':
    main()