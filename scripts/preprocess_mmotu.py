"""
MMOTU OTU-2D preprocessing for WAFS-Net.

This script converts the original MMOTU classification split
into the directory structure required for binary ovarian tumor
classification.

The original MMOTU images are NOT redistributed by this repository.

Expected raw structure:

data/raw/US_MMOTU/MMOTU/OTU_2d/
├── images/
├── train_cls.txt
└── val_cls.txt

Output:

data/processed/mmotu_2d_cls/
├── train/
│   ├── benign/
│   └── malignant/
└── val/
    ├── benign/
    └── malignant/
"""

import argparse
import shutil
from pathlib import Path


def parse_classification_file(txt_path):
    """
    Read the original MMOTU classification file.

    Expected line format:
        image_name class_label
    """

    samples = []

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            image_name = parts[0]
            class_label = int(parts[1])

            samples.append(
                (
                    image_name,
                    class_label,
                )
            )

    return samples


def build_binary_mapping():
    """
    Define the mapping from original MMOTU categories
    to binary diagnostic labels.

    IMPORTANT:
    The numerical IDs below must correspond to the class
    definitions in the original MMOTU release used in the study.

    Binary labels:
        0 = benign
        1 = malignant
    """

    # Fill/verify these IDs against the original MMOTU
    # class-definition file before reproducing the experiment.
    #
    # Example:
    #
    # original_to_binary = {
    #     <chocolate_cyst_id>: 0,
    #     <serous_cystadenoma_id>: 0,
    #     <teratoma_id>: 0,
    #     <theca_cell_tumor_id>: 0,
    #     <simple_cyst_id>: 0,
    #     <mucinous_cystadenoma_id>: 0,
    #     <high_grade_serous_carcinoma_id>: 1,
    # }

    original_to_binary = {}

    return original_to_binary


def prepare_split(
    raw_root,
    output_root,
    split,
    class_mapping,
):

    image_dir = (
        raw_root
        / "images"
    )

    classification_file = (
        raw_root
        / f"{split}_cls.txt"
    )

    if not classification_file.exists():
        raise FileNotFoundError(
            f"Classification file not found: "
            f"{classification_file}"
        )

    samples = parse_classification_file(
        classification_file
    )

    benign_count = 0
    malignant_count = 0
    skipped_count = 0

    for image_name, original_label in samples:

        if original_label not in class_mapping:

            skipped_count += 1
            continue

        binary_label = class_mapping[
            original_label
        ]

        if binary_label == 0:
            class_name = "benign"
            benign_count += 1

        elif binary_label == 1:
            class_name = "malignant"
            malignant_count += 1

        else:
            raise ValueError(
                "Binary labels must be 0 or 1."
            )

        source_path = (
            image_dir
            / image_name
        )

        destination_dir = (
            output_root
            / split
            / class_name
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        destination_path = (
            destination_dir
            / Path(image_name).name
        )

        if not source_path.exists():

            print(
                f"[WARNING] Missing image: "
                f"{source_path}"
            )

            continue

        shutil.copy2(
            source_path,
            destination_path,
        )

    print(
        f"\n{split.upper()}"
    )

    print(
        f"Benign    : {benign_count}"
    )

    print(
        f"Malignant : {malignant_count}"
    )

    print(
        f"Skipped   : {skipped_count}"
    )


def main(args):

    raw_root = Path(
        args.raw_root
    )

    output_root = Path(
        args.output_root
    )

    class_mapping = (
        build_binary_mapping()
    )

    if len(class_mapping) == 0:

        raise RuntimeError(
            "\nThe MMOTU original-class-to-binary mapping "
            "has not yet been specified.\n"
            "Please verify the numerical class IDs from the "
            "original MMOTU release and update "
            "build_binary_mapping() before running this script."
        )

    for split in [
        "train",
        "val",
    ]:

        prepare_split(
            raw_root=raw_root,
            output_root=output_root,
            split=split,
            class_mapping=class_mapping,
        )

    print(
        "\nMMOTU binary preprocessing completed."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Prepare MMOTU OTU-2D for "
            "WAFS-Net binary classification."
        )
    )

    parser.add_argument(
        "--raw_root",
        type=str,
        default=(
            "data/raw/"
            "US_MMOTU/MMOTU/OTU_2d"
        ),
    )

    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "data/processed/"
            "mmotu_2d_cls"
        ),
    )

    args = parser.parse_args()

    main(args)
