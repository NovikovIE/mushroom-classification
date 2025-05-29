import os

import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class MushroomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        self.images = []
        self.labels = []
        self.categories_labels = []

        categories = ["conditionally_edible", "deadly", "edible", "poisonous"]
        category_counter = 0

        for category_idx, category in enumerate(categories):
            category_path = os.path.join(root, category)

            for species_dir in os.listdir(category_path):
                species_path = os.path.join(category_path, species_dir)

                if os.path.isdir(species_path):
                    for img_file in os.listdir(species_path):
                        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                            img_path = os.path.join(species_path, img_file)
                            self.images.append(img_path)
                            self.labels.append(category_counter)

                        self.categories_labels.append(category_idx)

                category_counter += 1

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        category = self.categories_labels[index]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(0, len(self.images)))

        if self.transform:
            image = self.transform(image)

        return image, label, category


class MushroomDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        batch_size=32,
        num_workers=4,
        image_size=224,
        transform_mean=None,
        transform_std=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self.transform_mean = transform_mean
        self.transform_std = transform_std

        self.train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
                transforms.RandomRotation(degrees=10),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std),
            ]
        )

        self.class_to_category = self._generate_class_mapping()

    def _generate_class_mapping(self):
        categories = ["conditionally_edible", "deadly", "edible", "poisonous"]
        class_to_category = {}
        class_idx = 0

        data_dir_train = os.path.join(self.data_dir, "train")

        for category_idx, category in enumerate(categories):
            category_path = os.path.join(data_dir_train, category)
            species_dirs = [
                d
                for d in os.listdir(category_path)
                if os.path.isdir(os.path.join(category_path, d))
            ]

            for _ in species_dirs:
                class_to_category[class_idx] = category_idx
                class_idx += 1

        return class_to_category

    def get_class_to_category(self):
        return self.class_to_category

    def get_class_to_name(self):
        categories = ["conditionally_edible", "deadly", "edible", "poisonous"]
        class_to_name = {}
        class_idx = 0

        data_dir_train = os.path.join(self.data_dir, "train")

        for _, category in enumerate(categories):
            category_path = os.path.join(data_dir_train, category)
            species_dirs = [
                d
                for d in os.listdir(category_path)
                if os.path.isdir(os.path.join(category_path, d))
            ]

            for dir in species_dirs:
                class_to_name[class_idx] = dir
                class_idx += 1

        return class_to_name

    def setup(self, stage=None):
        self.train_dataset = MushroomDataset(
            os.path.join(self.data_dir, "train"), transform=self.train_transform
        )
        self.val_dataset = MushroomDataset(
            os.path.join(self.data_dir, "val"), transform=self.val_transform
        )
        self.test_dataset = MushroomDataset(
            os.path.join(self.data_dir, "test"), transform=self.val_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
