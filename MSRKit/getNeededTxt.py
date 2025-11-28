import os
import argparse
from collections import defaultdict

EXT = ".flac"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_dir",
        default="../data/MSRBench",
        help="Target Path",
    )
    parser.add_argument(
        "--enhanced_root",
        default="../data/output",
        help="enhanced_dir = enhanced_root / instru_type",
    )
    parser.add_argument(
        "--instru_type",
        default="Percussions",
        help="enhanced_dir, such as Drums/Bass/Vocals",
    )
    parser.add_argument(
        "--file_list",
        default="file_list.txt",
        help="Output path for file_list",
    )
    args = parser.parse_args()

    target_dir = args.target_dir + f"/{args.instru_type}/target"
    enhanced_dir = os.path.join(args.enhanced_root, args.instru_type)

    if not os.path.isdir(target_dir):
        raise FileNotFoundError(f"Target_dir not exist: {target_dir}")
    if not os.path.isdir(enhanced_dir):
        raise FileNotFoundError(f"Enhanced_dir not exist: {enhanced_dir}")

    enhanced_index = defaultdict(list)
    for fname in os.listdir(enhanced_dir):
        if not fname.lower().endswith(EXT):
            continue
        stem, _ = os.path.splitext(fname)
        if "_DT" in stem:
            base_stem = stem.split("_DT", 1)[0]   # "0"
        else:
            continue

        enhanced_index[base_stem].append(fname)

    lines = []
    missing = 0

    for fname in sorted(os.listdir(target_dir)):
        if not fname.lower().endswith(EXT):
            continue

        target_path = os.path.join(target_dir, fname)
        stem, _ = os.path.splitext(fname)

        matched_enhanced = enhanced_index.get(stem, [])
        if not matched_enhanced:
            missing += 1
            continue

        target_abs = os.path.abspath(target_path)
        for enh_fname in sorted(matched_enhanced):
            enhance_path = os.path.join(enhanced_dir, enh_fname)
            enhance_abs = os.path.abspath(enhance_path)
            lines.append(f"{target_abs}|{enhance_abs}")

    with open(args.file_list, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Write {len(lines)} to {args.file_list}")
    if missing > 0:
        print(f"Have missing file:{missing}")


if __name__ == "__main__":
    main()
