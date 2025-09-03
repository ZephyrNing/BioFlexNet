# src/training/dataset_imagenet.py

import json, os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision import transforms

class ImageNet100Dataset(Dataset):
    def __init__(self, folder: Path, mode: str = "TRAIN"):
        self.folder = Path(folder)
        self.mode = mode.upper()

        labels_file = self.folder / "Labels.json"
        labels_json = json.load(open(labels_file, "r"))


        self.class_to_idx = {
            synset: idx for idx, synset in enumerate(sorted(labels_json.keys()))
        }

        transform_list = [
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        self.transforms = transforms.Compose(transform_list)

        self.image_paths, self.image_labels = [], []
        self._gather()

    def _gather(self):
        folders = [f"train.X{i}" for i in range(1, 5)] if self.mode == "TRAIN" else ["val.X"]
        for folder_name in folders:
            path = self.folder / folder_name
            if not path.exists():
                continue
            for class_key in os.listdir(path):
                class_folder = path / class_key
                if class_folder.is_dir() and class_key in self.class_to_idx:
                    y = self.class_to_idx[class_key] 
                    for image_name in os.listdir(class_folder):
                        self.image_paths.append(class_folder / image_name)
                        self.image_labels.append(y)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = read_image(str(image_path)).float() / 255.0
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        if image.shape[0] == 4:
            image = image[:3, :, :]
        image = self.transforms(image)
        label = self.image_labels[idx]
        return image, label
