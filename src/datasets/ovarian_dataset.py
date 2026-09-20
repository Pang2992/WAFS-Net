"""
Dataset utilities for WAFS-Net.

Expected directory structure:

dataset_root/
├── train/
│   ├── benign/
│   └── malignant/
└── val/
    ├── benign/
    └── malignant/

Class labels:
    benign    -> 0
    malignant -> 1
"""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


class OvarianBinaryDataset(Dataset):

    def __init__(
        self,
        root_dir,
        transform=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform

        self.class_to_idx = {
            "benign": 0,
            "malignant": 1,
        }

        self.samples = []

        for class_name, label in self.class_to_idx.items():

            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                continue

            for path in sorted(class_dir.rglob("*")):

                if (
                    path.is_file()
                    and path.suffix.lower() in IMG_EXTENSIONS
                ):
                    self.samples.append(
                        (path, label)
                    )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No supported images were found in "
                f"{self.root_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_train_transform(
    image_size=224,
):
    """
    Training-time image preprocessing and augmentation.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.RandomHorizontalFlip(
                p=0.5
            ),

            transforms.RandomRotation(
                degrees=10
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def get_eval_transform(
    image_size=224,
):
    """
    Validation/test image preprocessing.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )


def build_dataset(
    root_dir,
    training=False,
    image_size=224,
):

    if training:
        transform = get_train_transform(
            image_size=image_size
        )
    else:
        transform = get_eval_transform(
            image_size=image_size
        )

    return OvarianBinaryDataset(
        root_dir=root_dir,
        transform=transform,
    )


if __name__ == "__main__":

    print(
        "OvarianBinaryDataset module "
        "for WAFS-Net."
    )
