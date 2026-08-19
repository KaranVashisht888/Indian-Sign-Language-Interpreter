"""
organize_include.py

Reorganizes AI4Bharat's INCLUDE dataset into the flat layout that
prepare_dataset.py expects.

INCLUDE ships as:
    <download_dir>/Adjectives/96. wet/MVI_5246.MOV
    <download_dir>/Greetings/55. Thank you/MVI_9985.MOV
    <download_dir>/People/61. Father/MVI_4940.MOV

prepare_dataset.py wants:
    data/raw_videos/wet/MVI_5246.MOV
    data/raw_videos/thank_you/MVI_9985.MOV
    data/raw_videos/father/MVI_4940.MOV

This script strips the numeric prefixes ("96. wet" -> "wet"), normalizes
names to lowercase_with_underscores, flattens away the category level,
and copies (or symlinks) only the words you asked for.

Getting the dataset first:
    The videos live on Zenodo (record 4010759) — see the download script
    on https://huggingface.co/datasets/ai4bharat/INCLUDE. It's CC-BY-4.0,
    no access request needed. It's a large download; if you only want a
    starting subset, you can grab individual category zips from the Zenodo
    record page rather than all of them.

Usage:
    # Start small — just these 5 words:
    python src/organize_include.py --include-dir /path/to/INCLUDE

    # A custom word list:
    python src/organize_include.py --include-dir /path/to/INCLUDE \\
        --words father mother house school water

    # Everything it can find (not recommended as a first run):
    python src/organize_include.py --include-dir /path/to/INCLUDE --all

    # Symlink instead of copy, to save disk space (Linux/macOS):
    python src/organize_include.py --include-dir /path/to/INCLUDE --symlink

Then:
    python src/prepare_dataset.py
    python src/train.py
"""

import argparse
import re
import shutil
from pathlib import Path

# A deliberately tiny default subset. Get the whole pipeline working
# end-to-end on 5 words before scaling up — a working 5-word demo beats a
# half-configured 50-word one, and scaling up is just more data through
# the same scripts.
DEFAULT_WORDS = ["father", "mother", "house", "school", "loud"]

# INCLUDE uses .MOV and .MP4 (uppercase). Matching is case-insensitive
# below, so don't worry about the exact casing on your filesystem.
VIDEO_SUFFIXES = {".mov", ".mp4", ".avi", ".mkv"}


def normalize_word(folder_name: str) -> str:
    """
    '96. wet'       -> 'wet'
    '55. Thank you' -> 'thank_you'
    '1. Dog'        -> 'dog'
    """
    without_prefix = re.sub(r"^\s*\d+\.\s*", "", folder_name)
    return without_prefix.strip().lower().replace(" ", "_")


def find_word_dirs(include_dir: Path) -> dict[str, list[Path]]:
    """
    Walks the INCLUDE tree and maps normalized word -> list of source
    folders holding that word's clips. It's a list because the same word
    can legitimately appear under more than one category folder.
    """
    word_dirs: dict[str, list[Path]] = {}

    for category_dir in sorted(p for p in include_dir.iterdir() if p.is_dir()):
        for word_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            word = normalize_word(word_dir.name)
            word_dirs.setdefault(word, []).append(word_dir)

    return word_dirs


def organize(include_dir: Path, output_dir: Path, words: list[str] | None, use_symlink: bool) -> None:
    if not include_dir.exists():
        raise FileNotFoundError(f"{include_dir} does not exist — check --include-dir.")

    word_dirs = find_word_dirs(include_dir)
    if not word_dirs:
        raise ValueError(
            f"No category/word folders found under {include_dir}. Expected a layout like "
            f"'{include_dir}/Adjectives/96. wet/MVI_5246.MOV' — point --include-dir at the "
            f"folder that directly contains the category folders."
        )

    print(f"Found {len(word_dirs)} distinct words in {include_dir}\n")

    if words is None:
        selected = sorted(word_dirs)
    else:
        selected = [w.lower().replace(" ", "_") for w in words]
        missing = [w for w in selected if w not in word_dirs]
        if missing:
            available_sample = ", ".join(sorted(word_dirs)[:25])
            raise ValueError(
                f"These requested words weren't found: {missing}\n"
                f"Sample of what IS available: {available_sample} ...\n"
                f"(Run with --list to print every available word.)"
            )

    total_clips = 0
    for word in selected:
        dest_dir = output_dir / word
        dest_dir.mkdir(parents=True, exist_ok=True)

        clips = [
            clip
            for source_dir in word_dirs[word]
            for clip in sorted(source_dir.iterdir())
            if clip.suffix.lower() in VIDEO_SUFFIXES
        ]

        if not clips:
            print(f"  {word}: no video files found, skipping")
            continue

        for clip in clips:
            dest = dest_dir / clip.name
            if dest.exists():
                continue
            if use_symlink:
                dest.symlink_to(clip.resolve())
            else:
                shutil.copy2(clip, dest)

        total_clips += len(clips)
        print(f"  {word}: {len(clips)} clips")

    print(f"\nDone. {total_clips} clips across {len(selected)} words -> {output_dir}")
    print("Next: python src/prepare_dataset.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reorganize INCLUDE into data/raw_videos/<word>/")
    parser.add_argument("--include-dir", required=True, type=Path,
                        help="Path to the downloaded INCLUDE folder (the one containing Adjectives/, Greetings/, etc)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw_videos"))
    parser.add_argument("--words", nargs="+", default=None,
                        help=f"Words to include. Default: {' '.join(DEFAULT_WORDS)}")
    parser.add_argument("--all", action="store_true", help="Use every word found (large — not a good first run)")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying, to save disk space")
    parser.add_argument("--list", action="store_true", help="Just print every available word and exit")
    args = parser.parse_args()

    if args.list:
        for word in sorted(find_word_dirs(args.include_dir)):
            print(word)
        return

    words = None if args.all else (args.words or DEFAULT_WORDS)
    organize(args.include_dir, args.output_dir, words, args.symlink)


if __name__ == "__main__":
    main()
