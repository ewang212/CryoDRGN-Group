import numpy as np
import pickle
import torchvision.transforms as transforms
import cv2
from PIL import Image
from torch.utils.data import Dataset
from utils import load_config, hartley_transform
from torch.utils.data import random_split
from torch.utils.data import DataLoader


class Data:
    def __init__(self, config_file_name):
        self.config = load_config(config_folder="config", config_file=config_file_name, config_name="default")
        self.train_dataset = None
        self.test_dataset = None
        
    
    def set_cryo_data(self): 
        data = self.config.data
        projections_file = open(data.projections_file_path, 'rb')
        rotations_data_file = open(data.rotations_file_path, 'rb')
        projections = pickle.load(projections_file)
        rotations = pickle.load(rotations_data_file)
        images_fourier = []
        
        interpolation = cv2.INTER_LINEAR if data.inter_linear else cv2.INTER_NEAREST

        for projection, rot in zip(projections, rotations): 
            data2D = cv2.resize(projection, (data.D, data.D), interpolation=interpolation)
            hartley = hartley_transform(data2D)
            images_fourier.append(hartley)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = CryoBenchDataset(images=images_fourier, transform=transform, rotation=rotations)
        
        train_dataset, test_dataset = random_split(dataset, [data.train_pctg, 1-data.train_pctg])
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


    def get_training_loader(self):
        if self.train_dataset is None or self.test_dataset is None:
            self.set_cryo_data()

        data = self.config.data
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=data.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        return train_loader

    def get_validation_loader(self):
        if self.train_dataset is None or self.test_dataset is None:
            self.set_cryo_data()

        data = self.config.data
        validation_loader = DataLoader(
            self.test_dataset,
            batch_size=data.batch_size,
            shuffle=False, 
            num_workers=0,
            drop_last=True
        )
        return validation_loader


class CryoBenchDataset(Dataset):
    def __init__(self, images, rotation = None, transform = None):
        self.images = images
        self.transform = transform
        self.rotation = rotation
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)
        
        phi = self.rotation[idx]

        return image, phi

