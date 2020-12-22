from torch.utils.data.dataset import T_co
from torchvision import transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import rawpy
class MnistDataLoader(BaseDataLoader):
    """
    Data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, noise, shuffle=True, validation_split=0.0, num_workers=1, crop_size=256):
        self.data_dir = data_dir
        self.dataset = ImageDataset(crop_size, noise)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ImageDataset(Dataset):

    def __init__(self, crop_size:int, noise:str):
        super().__init__()
        self.trsfms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(crop_size)
        ])

        self.train_list:pd.DataFrame = pd.read_csv()
        self.val_list:pd.DataFrame = pd.read_csv()
        self.main_list:pd.DataFrame = pd.concat([self.train_list, self.val_list], axis=0)
        self.noise = noise
    def __len__(self) -> int:
        return self.train_list.shape[0] + self.val_list.shape[0]

    def __getitem__(self, index) -> T_co:
        clean = rawpy.imread(self.main_list[index][0]).postprocess()
        noisy = rawpy.imread(self.main_list[index][1]).postprocess()
        clean = self.trsfms(clean)
        noisy = self.trsfms(noisy)
        return clean, noisy