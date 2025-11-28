import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import soundfile as sf
import numpy as np
from tqdm import tqdm

from models import MelRNN, MelRoFormer, UNet


def load_generator(config: Dict[str, Any], checkpoint_path: Path, device: str = 'cuda') -> nn.Module:
    """Initialize and load the generator model from Lightning ckpt or raw state_dict."""
    model_cfg = config['model']

    if model_cfg['name'] == 'MelRNN':
        generator = MelRNN.MelRNN(**model_cfg['params'])
    elif model_cfg['name'] == 'MelRoFormer':
        generator = MelRoFormer.MelRoFormer(**model_cfg['params'])
    elif model_cfg['name'] == 'MelUNet':
        generator = UNet.MelUNet(**model_cfg['params'])
    else:
        raise ValueError(f"Unknown model name: {model_cfg['name']}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        full_sd = ckpt["state_dict"]
        state_dict = {
            k.replace("generator.", ""): v
            for k, v in full_sd.items()
            if k.startswith("generator.")
        }
        print(f"[Load] Loaded Lightning ckpt from {checkpoint_path}, "
              f"extracted {len(state_dict)} generator params.")
    else:
        state_dict = ckpt
        print(f"[Load] Loaded raw state_dict from {checkpoint_path} with {len(state_dict)} keys.")

    generator.load_state_dict(state_dict)
    generator = generator.to(device)
    generator.eval()
    return generator


def process_audio(audio: np.ndarray, generator: nn.Module, device: str = 'cuda') -> np.ndarray:
    """Process a single audio array through the generator."""
    # Convert to tensor: (channels, samples) -> (1, channels, samples)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # Add channel dimension for mono
    
    audio_tensor = torch.from_numpy(audio).float().to(device)
    
    # Run inference
    with torch.no_grad():
        output_tensor = generator(audio_tensor)
    
    # Convert back to numpy: (1, channels, samples) -> (channels, samples)
    output_audio = output_tensor.cpu().numpy()
    
    return output_audio


def load_audio_channels_first(path: Path):
    """Load audio and return (channels, samples), sr."""
    audio, sr = sf.read(path)
    # soundfile: mono -> (T,), stereo -> (T, C)
    if audio.ndim == 1:
        audio_cf = audio[np.newaxis, :]  # (1, T)
    else:
        audio_cf = audio.T  # (C, T)
    return audio_cf, sr


def is_silent(audio: np.ndarray, threshold: float) -> bool:
    if audio.size == 0:
        return True
    max_abs = np.max(np.abs(audio))
    return max_abs < threshold


def group_files_by_instrument(input_dir: Path, audio_files: List[Path]):
    groups: Dict[str, List[Path]] = {}
    for p in audio_files:
        rel_parts = p.relative_to(input_dir).parts
        if len(rel_parts) < 2:
            instrument = "unknown"
        else:
            instrument = rel_parts[0]
        groups.setdefault(instrument, []).append(p)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Run inference on audio files using trained generator (per-instrument checkpoints)")
    parser.add_argument("--config", type=str, default="../pretrain/baseline/config.yaml")
    parser.add_argument("--checkpoint_dir", type=str, default="../pretrain/baseline",
                        help="Directory containing {instrument}.pth checkpoints")
    parser.add_argument("--input_dir", type=str, default="../AoMSS/BSR_output",
                        help="Directory with structure input_dir/{instrument_name}/*.flac")
    parser.add_argument("--original_dir", type=str, default="../data/OrganizersMixture",
                        help="Original audio dir with same structure as input_dir for silent replacement")
    parser.add_argument("--output_dir", type=str, default="../CUPAudioGroup",
                        help="Output dir, will mirror the subdirectory structure of input_dir")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--silence_threshold", type=float, default=1e-4,
                        help="Threshold for detecting almost-silent audio (on max abs amplitude)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup paths
    input_dir = Path(args.input_dir)
    original_dir = Path(args.original_dir)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(input_dir.rglob("*.flac"))

    if len(audio_files) == 0:
        print(f"No .flac files found in {input_dir}")
        return

    print(f"Found {len(audio_files)} audio files under {input_dir}")

    groups = group_files_by_instrument(input_dir, audio_files)
    print(f"Detected instruments: {list(groups.keys())}")

    device = args.device

    for instrument, files in groups.items():
        skip_instruments = ["Percussions"]
        if instrument in skip_instruments:
            continue
        if instrument == "unknown":
            print("[Warning] Some files are not under input_dir/{instrument}/...; these are grouped as 'unknown'.")

        checkpoint_path = checkpoint_dir / f"{instrument}.pth"

        if not checkpoint_path.exists():
            print(f"[Warning] Checkpoint file not found for instrument '{instrument}': {checkpoint_path}. "
                  f"Skipping all files of this instrument.")
            continue

        print(f"\n===== Processing instrument: {instrument} =====")
        print(f"Using checkpoint: {checkpoint_path}")

        with torch.no_grad():
            generator = load_generator(config, checkpoint_path, device=device)
            print(f"Model for '{instrument}' loaded successfully.")

            for audio_file in tqdm(files, desc=f"Processing {instrument}"):
                rel_path = audio_file.relative_to(input_dir)  # instrument/xxx.flac
                orig_path = original_dir / rel_path
                out_path = output_dir / rel_path

                out_path.parent.mkdir(parents=True, exist_ok=True)

                try:
                    audio_cf, sr = load_audio_channels_first(audio_file)
                except Exception as e:
                    print(f"[Error] Failed to load input file {audio_file}: {e}")
                    continue

                if is_silent(audio_cf, threshold=args.silence_threshold):
                    if orig_path.exists():
                        try:
                            orig_audio_cf, orig_sr = load_audio_channels_first(orig_path)
                            if orig_sr != sr:
                                print(f"[Warning] SR mismatch for {rel_path}: "
                                      f"input {sr}, original {orig_sr}. Using original SR.")
                                sr = orig_sr
                            audio_cf = orig_audio_cf
                            print(f"[Replace] {rel_path} is almost silent. Using original: {orig_path}")
                        except Exception as e:
                            print(f"[Error] Failed to load original file {orig_path}: {e}. "
                                  f"Fallback to silent input.")
                    else:
                        print(f"[Warning] {rel_path} is almost silent, but original file not found at {orig_path}. "
                              f"Using current (silent) audio as input.")

                output_audio_cf = process_audio(audio_cf, generator, device=device)

                if output_audio_cf.ndim == 1:
                    output_audio = output_audio_cf  # (T,)
                else:
                    output_audio = output_audio_cf.T  # (T, C)

                try:
                    sf.write(out_path, output_audio, sr)
                except Exception as e:
                    print(f"[Error] Failed to save output file {out_path}: {e}")

    print(f"\nAll done! Output saved under {output_dir}")


if __name__ == '__main__':
    main()
