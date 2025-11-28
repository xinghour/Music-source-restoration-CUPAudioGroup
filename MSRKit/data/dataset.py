from pathlib import Path
import random
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import List, Optional, Dict, Union, Tuple, Any
from torch.utils.data import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = ['.flac', '.mp3', '.wav']

def fix_length_to_duration(target: np.ndarray, duration: float, sr: int) -> np.ndarray:
    target_length = target.shape[-1]
    required_length = int(duration * sr)
    if target_length < required_length:
        return np.pad(target, ((0, 0), (0, required_length - target_length)), mode='constant')
    if target_length > required_length:
        return target[:, :required_length]
    return target

def load_audio(file_path: Path, offset: float, duration: float, sr: int) -> np.ndarray:
    try:
        audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=duration, mono=False)
        if len(audio.shape) == 1: audio = audio.reshape(1, -1)
        if audio.shape[1] == 0: return np.zeros((2, int(sr * duration)))
        if audio.shape[0] == 1: audio = np.vstack([audio, audio])
        return audio
    except Exception as e:
        logger.error(f"Error loading {file_path} at offset {offset}: {e}")
        return np.zeros((2, int(sr * duration)))

class RawStemsWithDT(Dataset):
    def __init__(
        self,
        root_directory: Union[str, Path],
        sr: int = 48000,
        clip_duration: float = 4.0,
        max_samples: Optional[int] = None
    ) -> None:
        self.root_directory = Path(root_directory)
        self.sr = sr
        self.clip_duration = clip_duration
        self.max_samples = max_samples

        self.target_dir = self.root_directory / "target"
        self.voc_root = self.root_directory / "Voc_sample_100"

        if not self.target_dir.exists():
            raise FileNotFoundError(f"Target directory not found: {self.target_dir}")
        if not self.voc_root.exists():
            raise FileNotFoundError(f"Voc_sample_100 directory not found: {self.voc_root}")

        self.mixture_subdirs: List[Path] = [
            d for d in self.voc_root.iterdir()
            if d.is_dir() and d.name.startswith("sep_DT")
        ]
        if not self.mixture_subdirs:
            raise FileNotFoundError(f"No sep_DT* folders found under {self.voc_root}")

        logger.info(f"Found mixture dirs: {[d.name for d in self.mixture_subdirs]}")

        samples = self._index_pairs()

        if self.max_samples is not None and self.max_samples < len(samples):
            random.shuffle(samples)
            samples = samples[: self.max_samples]
            logger.info(f"Using {len(samples)} training samples (max_samples={self.max_samples})")

        self.samples = samples

        if not self.samples:
            raise ValueError("No (target, mixture) pairs found. Check training root layout.")

        logger.info(f"Final number of training samples: {len(self.samples)}")

    def _index_pairs(self) -> List[Dict[str, Any]]:
        samples = []

        target_files = sorted(
            [p for p in self.target_dir.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS]
        )

        target_map = {}
        for t_path in target_files:
            stem_no_ext = t_path.stem
            if not stem_no_ext.endswith("_target"):
                continue
            base_id = stem_no_ext[:-len("_target")]
            target_map[base_id] = t_path

        for sep_dir in self.mixture_subdirs:
            for p in sep_dir.iterdir():
                if p.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                stem_no_ext = p.stem  # e.g. mskrkit_Voc_000000_mixture
                if not stem_no_ext.endswith("_mixture"):
                    continue

                base_id = stem_no_ext[:-len("_mixture")]
                if base_id not in target_map:
                    continue

                samples.append({
                    "base_id": base_id,
                    "target": target_map[base_id],
                    "mixture": p
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        target_path: Path = sample["target"]
        mixture_path: Path = sample["mixture"]

        target = load_audio(target_path, 0.0, self.clip_duration, self.sr)
        mixture = load_audio(mixture_path, 0.0, self.clip_duration, self.sr)

        target = fix_length_to_duration(target, self.clip_duration, self.sr)
        mixture = fix_length_to_duration(mixture, self.clip_duration, self.sr)

        max_val = max(
            np.max(np.abs(target)),
            np.max(np.abs(mixture)),
        ) + 1e-8
        target = target / max_val
        mixture = mixture / max_val

        return {
            "target": np.nan_to_num(target),
            "mixture": np.nan_to_num(mixture),
        }

class MSRBenchVal(Dataset):
    def __init__(
        self,
        root_directory: Union[str, Path],
        sr: int = 48000,
        clip_duration: float = 4.0,
        n_per_dt: int = 10,
    ) -> None:

        self.root = Path(root_directory)
        self.sr = sr
        self.clip_duration = clip_duration
        self.n_per_dt = n_per_dt

        voc = self.root / "Vocals"
        self.target_dir = voc / "target"

        if not self.target_dir.exists():
            raise FileNotFoundError(self.target_dir)

        # id -> target path
        self.targets = {
            p.stem: p
            for p in self.target_dir.iterdir()
            if p.suffix.lower() in AUDIO_EXTENSIONS
        }

        # mixture by DT folders
        self.samples = []

        for dt_dir in sorted((voc.iterdir())):
            if not dt_dir.is_dir() or dt_dir.name == "target":
                continue

            dt_type = dt_dir.name  # "DT0", "DT1", ...

            items = []
            for p in dt_dir.iterdir():
                if p.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue

                stem = p.stem       # "0_DT1"
                id_part, _ = stem.split("_")  # "0"

                if id_part not in self.targets:
                    continue

                items.append({
                    "id": id_part,
                    "target": self.targets[id_part],
                    "mixture": p,
                    "dt_type": dt_type,
                })

            random.shuffle(items)
            self.samples.extend(items[: self.n_per_dt])

        logger.info(f"Validation samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        target = load_audio(s["target"], 0.0, self.clip_duration, self.sr)
        mixture = load_audio(s["mixture"], 0.0, self.clip_duration, self.sr)

        target = fix_length_to_duration(target, self.clip_duration, self.sr)
        mixture = fix_length_to_duration(mixture, self.clip_duration, self.sr)

        m = max(np.max(np.abs(target)), np.max(np.abs(mixture))) + 1e-8
        target = target / m
        mixture = mixture / m

        return {
            "mixture": mixture,
            "target": target,
            "dt_type": s["dt_type"],
            "id": s["id"],
        }
