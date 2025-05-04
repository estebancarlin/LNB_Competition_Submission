import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.params import (
    S1_PATH, S2_PATH, MASK_PATH,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    TRAINING_BATCH_SIZE, VALIDATION_BATCH_SIZE, TESTING_BATCH_SIZE,
    SHUFFLE_DATA
)
from src.traitement import (
    find_previous, modif_path, load_image,
    normalisation_s1, normalisation_s2, moyenne_s2
)

# Set seed for reproducibility
np.random.seed(0)


class DataGenerator(Dataset):
    """Dataset for training and evaluation."""

    def __init__(self, names):
        self.names = names

    def __len__(self):
        return 2000

    def __getitem__(self, index):
        name_0, name_1 = find_previous(self.names, index)

        # Build paths
        s2_paths = [
            modif_path(os.path.join(S2_PATH, name)) for name in [name_0, name_1, self.names[index]]
        ]
        s1_path = modif_path(os.path.join(S1_PATH, self.names[index]))
        mask_paths = [
            modif_path(os.path.join(MASK_PATH, name)) for name in [name_0, name_1, self.names[index]]
        ]

        # Load masks with fallback
        def safe_load_mask(path):
            try:
                return load_image(path)
            except Exception:
                return torch.ones((256, 256)) * 4

        masks = [safe_load_mask(path) for path in mask_paths]

        # Load and normalize images
        s1 = normalisation_s1(load_image(s1_path).clone().detach())
        s2_images = [load_image(p) for p in s2_paths]
        s2_1 = normalisation_s2(s2_images[1].clone().detach().unsqueeze(0))
        s2_2 = normalisation_s2(s2_images[2].clone().detach().unsqueeze(0))

        # Compute normalized average of past S2 with masks
        mean_s2 = normalisation_s2(torch.tensor(moyenne_s2(s2_images[0], s2_images[1], masks[0], masks[1])))

        X = torch.cat((mean_s2, s1), dim=0)
        Y = torch.cat((s2_2, masks[2].unsqueeze(0)), dim=0)

        return X, Y


def data_split(names):
    """Split the dataset into train, val, and test based on size ratios."""
    n = len(names)
    train_end = int(TRAIN_SPLIT * n)
    val_end = train_end + int(VAL_SPLIT * n)
    return names[:train_end], names[train_end:val_end], names[val_end:]


def create_generators():
    """Create PyTorch dataloaders for training, validation, and testing."""
    names = os.listdir(S2_PATH)
    train_data, val_data, test_data = data_split(names)

    train_loader = DataLoader(DataGenerator(train_data), batch_size=TRAINING_BATCH_SIZE,
                              shuffle=SHUFFLE_DATA, drop_last=True)
    val_loader = DataLoader(DataGenerator(val_data), batch_size=VALIDATION_BATCH_SIZE,
                            shuffle=SHUFFLE_DATA, drop_last=True)
    test_loader = DataLoader(DataGenerator(test_data), batch_size=TESTING_BATCH_SIZE,
                             shuffle=SHUFFLE_DATA, drop_last=True)

    return train_loader, val_loader, test_loader


class PredictGenerator(Dataset):
    """Dataset for inference using a CSV file."""

    def __init__(self, triplets, root_path):
        self.triplets = triplets
        self.root_path = root_path
        self.S1 = os.path.join(root_path, 's1')
        self.S2 = os.path.join(root_path, 's2')
        self.MASK = os.path.join(root_path, 's2-mask')

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        name_0, name_1, name_2 = self.triplets[index]

        # Paths
        path_s1 = modif_path(os.path.join(self.S1, name_2))
        path_s2_0 = modif_path(os.path.join(self.S2, name_0))
        path_s2_1 = modif_path(os.path.join(self.S2, name_1))
        path_mask_0 = modif_path(os.path.join(self.MASK, name_0))
        path_mask_1 = modif_path(os.path.join(self.MASK, name_1))

        # Load images
        s1 = normalisation_s1(load_image(path_s1).clone().detach())
        s2_0 = load_image(path_s2_0)
        s2_1 = normalisation_s2(load_image(path_s2_1).clone().detach())
        mask_0 = load_image(path_mask_0)
        mask_1 = load_image(path_mask_1)

        # Mean S2 normalization
        mean_s2 = normalisation_s2(torch.tensor(moyenne_s2(s2_0, s2_1, mask_0, mask_1)).unsqueeze(0))

        X = torch.cat((mean_s2, s1), dim=0)
        Y = mean_s2  # target is just the mean_s2 for prediction

        return X, Y, name_2


def create_prediction(csv_path):
    """Create a prediction dataloader from a CSV path."""
    with open(csv_path, 'r') as file:
        reader = csv.reader(file, delimiter=",")
        triplets = [
            [os.path.basename(path) for path in row]
            for row in list(reader)[1:]
        ]
    dataset_root = os.path.dirname(csv_path)
    dataloader = DataLoader(PredictGenerator(triplets, dataset_root), batch_size=1)
    return dataloader