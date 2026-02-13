#!/usr/bin/env python3
"""Build a submission zip from the contents of a folder (e.g. submission_contents).

Zips the contents of the input folder, not the folder itself, so files appear
at the root of the archive (as Codabench expects for code submissions).

Usage (from starting kit directory):
  python utils/build_submission.py
  python utils/build_submission.py -i submission_contents -o my_submission.zip
"""

import argparse
import zipfile
from pathlib import Path


def build_submission(input_dir: Path, output_path: Path) -> None:
    """Zip the contents of input_dir into output_path (folder itself excluded)."""
    input_dir = input_dir.resolve()
    output_path = output_path.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in input_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(input_dir)
                zf.write(file_path, arcname)

    print(f"Created {output_path} from {input_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a submission zip from folder contents (e.g. submission_contents)."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("submission_contents"),
        help="Input folder to zip (default: submission_contents)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("submission.zip"),
        help="Output zip file path (default: submission.zip)",
    )
    args = parser.parse_args()
    build_submission(args.input, args.output)


if __name__ == "__main__":
    main()
