from torch.utils.data.dataset import T_co
from tqdm import tqdm
from base import BaseDataLoader
from torch.utils.data import Dataset
import os
import cv2
import shutil
import numpy as np
from torchvision import transforms
from torchvision.transforms import ToTensor


class dataloader(BaseDataLoader):

    def __init__(self, data_dir, batch_size, process_dataset, shuffle=True, validation_split=0.0, num_workers=1,
                 crop_size=256):
        self.data_dir = data_dir
        self.dataset = ImageDataset(data_dir, crop_size, process_dataset)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


def process_data(data_dir, crop_size):
    path = "Data/dataset_{}".format(crop_size)
    try:
        os.mkdir(path)
    except:
        pass
    num = 0
    for folder in tqdm(os.listdir(data_dir)):
        clean = cv2.imread(os.path.join(data_dir, folder, "GT_SRGB_010.PNG"))
        noisy = cv2.imread(os.path.join(data_dir, folder, "NOISY_SRGB_010.PNG"))
        assert clean.shape == noisy.shape
        for i in range(clean.shape[0] // crop_size):
            for j in range(clean.shape[1] // crop_size):
                clean_crop = clean[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size, :]
                noisy_crop = noisy[i * crop_size:(i + 1) * crop_size, j * crop_size:(j + 1) * crop_size, :]
                final = np.stack([clean_crop, noisy_crop])
                np.save(os.path.join(path, "{:06d}.npy".format(num)), final)
                num += 1
    return path, num


class ImageDataset(Dataset):

    def __init__(self, data_dir: str, crop_size: int, process_dataset: bool):
        super().__init__()

        if process_dataset:
            self.data_dir, self.num = process_data(data_dir, crop_size)
        else:
            self.data_dir = "Data/dataset_{}".format(crop_size)
            self.num = len(os.listdir(self.data_dir))
        self.data_list = os.listdir(self.data_dir)
        self.transforms = transforms.Compose([
            ToTensor()
        ])

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, index) -> T_co:
        img = np.load(os.path.join(self.data_dir, self.data_list[index]))
        clean = self.transforms(img[0, :, :, :])
        noisy = self.transforms(img[1, :, :, :])
        return clean, noisy
