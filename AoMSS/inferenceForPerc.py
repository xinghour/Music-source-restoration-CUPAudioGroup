# coding: utf-8
__author__ = 'Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/'

import time
import librosa
import sys
import os
import glob
import torch
import soundfile as sf
import numpy as np
from tqdm.auto import tqdm
import torch.nn as nn

# Using the embedded version of Python can also correctly import the utils module.
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.audio_utils import normalize_audio, denormalize_audio, draw_spectrogram
from utils.settings import get_model_from_config, parse_args_inference
from utils.model_utils import demix
from utils.model_utils import prefer_target_instrument, apply_tta, load_start_checkpoint

import warnings

warnings.filterwarnings("ignore")


def run_folder(model, args, config, device, verbose: bool = False):

    start_time = time.time()
    model.eval()

    # ① 递归地获取 input_folder 下所有 .flac 文件
    pattern = os.path.join(args.input_folder, '**', '*.flac')
    mixture_paths = sorted(glob.glob(pattern, recursive=True))

    sample_rate = getattr(config.audio, 'sample_rate', 44100)

    print(f"Looking for files with pattern: {pattern}")
    print(f"Total .flac files found: {len(mixture_paths)}. Using model sample rate: {sample_rate}")

    # ② 只保留 drums（如果需要改目标乐器，只改这一行）
    all_instruments = prefer_target_instrument(config)[:]
    target_instr_name = "hh"

    instruments = [instr for instr in all_instruments if instr.lower() == target_instr_name]

    if not instruments:
        print(f"WARNING: Target instrument '{target_instr_name}' not found in model instruments: {all_instruments}")
        print("         Will save all instruments instead.")
        instruments = all_instruments
    else:
        print(f"Model instruments: {all_instruments}")
        print(f"Will only save instrument(s): {instruments}")

    os.makedirs(args.store_dir, exist_ok=True)

    if not verbose:
        mixture_paths = tqdm(mixture_paths, desc="Total progress")

    detailed_pbar = not args.disable_detailed_pbar

    for path in mixture_paths:
        print(f"Processing track: {path}")

        try:
            mix, orig_sr = librosa.load(path, sr=None, mono=False)
        except Exception as e:
            print(f'Cannot read track: {format(path)}')
            print(f'Error message: {str(e)}')
            continue

        if orig_sr != sample_rate:
            print(f"Resampling from {orig_sr} Hz to {sample_rate} Hz for model inference...")
            mix = librosa.resample(mix, orig_sr=orig_sr, target_sr=sample_rate, axis=-1)
            sr = sample_rate
        else:
            sr = orig_sr

        if len(mix.shape) == 1:
            mix = np.expand_dims(mix, axis=0)  # (1, samples)
            if 'num_channels' in config.audio:
                if config.audio['num_channels'] == 2:
                    print('Convert mono track to stereo...')
                    mix = np.concatenate([mix, mix], axis=0)  # (2, samples)

        mix_orig = mix.copy()

        if 'normalize' in config.inference and config.inference['normalize'] is True:
            mix, norm_params = normalize_audio(mix)
        else:
            norm_params = None

        waveforms_orig = demix(
            config,
            model,
            mix,
            device,
            model_type=args.model_type,
            pbar=detailed_pbar
        )

        if args.use_tta:
            waveforms_orig = apply_tta(
                config,
                model,
                mix,
                waveforms_orig,
                device,
                args.model_type
            )

        if args.extract_instrumental:
            instr_main = 'vocals' if 'vocals' in waveforms_orig else next(iter(waveforms_orig.keys()))
            waveforms_orig['instrumental'] = mix_orig - waveforms_orig[instr_main]

        # 原文件名（不带扩展名），例如 "0_DT6"
        file_name = os.path.splitext(os.path.basename(path))[0]

        for instr in instruments:
            if instr not in waveforms_orig:
                print(f"Instrument '{instr}' not found in model output. Available: {list(waveforms_orig.keys())}")
                continue

            estimates = waveforms_orig[instr]

            if norm_params is not None:
                estimates = denormalize_audio(estimates, norm_params)

            if orig_sr != sr:
                print(f"Resampling {instr} from {sr} Hz back to original {orig_sr} Hz for saving...")
                estimates_for_save = librosa.resample(
                    estimates,
                    orig_sr=sr,
                    target_sr=orig_sr,
                    axis=-1
                )
            else:
                estimates_for_save = estimates

            # 这里继续使用 flac 保存
            use_flac = True
            if use_flac:
                codec = 'flac'
                subtype = 'PCM_16'
            else:
                codec = 'wav'
                subtype = getattr(args, 'pcm_type', 'PCM_16')

            # ③ 保留相对目录结构：input_folder 下的相对路径，映射到 store_dir
            rel_path = os.path.relpath(path, args.input_folder)        # e.g. "subdir/0_DT6.flac"
            rel_stem, _ = os.path.splitext(rel_path)                   # "subdir/0_DT6"
            output_path = os.path.join(args.store_dir, rel_stem + f".{codec}")

            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)

            sf.write(output_path, estimates_for_save.T, orig_sr, subtype=subtype)
            print("Wrote file:", output_path)

            if args.draw_spectro > 0:
                img_stem, _ = os.path.splitext(output_path)
                output_img_path = img_stem + ".jpg"
                draw_spectrogram(estimates_for_save.T, orig_sr, args.draw_spectro, output_img_path)
                print("Wrote file:", output_img_path)

    print(f"Elapsed time: {time.time() - start_time:.2f} seconds.")


def format_filename(template, **kwargs):
    result = template
    for k, v in kwargs.items():
        result = result.replace(f"{{{k}}}", str(v))
    *dirnames, fname = result.split("/")
    return dirnames, fname


def proc_folder(dict_args):
    args = parse_args_inference(dict_args)
    device = "cpu"
    if args.force_cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        print('CUDA is available, use --force_cpu to disable it.')
        device = f'cuda:{args.device_ids[0]}' if isinstance(args.device_ids, list) else f'cuda:{args.device_ids}'
    elif torch.backends.mps.is_available():
        device = "mps"

    print("Using device: ", device)

    model_load_start_time = time.time()
    torch.backends.cudnn.benchmark = True

    model, config = get_model_from_config("mdx23c", args.config_path)
    if 'model_type' in config.training:
        args.model_type = config.training.model_type
    if args.start_check_point:
        checkpoint = torch.load(args.start_check_point, weights_only=False, map_location='cpu')
        load_start_checkpoint(args, model, checkpoint, type_='inference')

    print("Instruments: {}".format(config.training.instruments))

    # in case multiple CUDA GPUs are used and --device_ids arg is passed
    if isinstance(args.device_ids, list) and len(args.device_ids) > 1 and not args.force_cpu:
        model = nn.DataParallel(model, device_ids=args.device_ids)

    model = model.to(device)

    print("Model load time: {:.2f} sec".format(time.time() - model_load_start_time))

    run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    proc_folder(None)
