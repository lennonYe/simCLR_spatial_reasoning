import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
class CustomImagePairDataset(Dataset):
    def __init__(self, root_folder,label_dir,transform=None):
        self.root_folder = root_folder
        self.image_list = os.listdir(root_folder)
        self.transform = transform
        self.labels = pd.read_csv(label_dir, names=['image_1', 'image_2', 'label'])
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        # img1_path = os.path.join(self.root_folder, self.image_list[idx])
        # img2_path = os.path.join(self.root_folder, self.image_list[(idx + 1) % len(self.image_list)])
        image_1 = self.labels.iloc[idx]['image_1']
        image_2 = self.labels.iloc[idx]['image_2']
        img1_path = os.path.join(self.root_folder,image_1)
        img2_path = os.path.join(self.root_folder,image_2)
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        label = self.labels.iloc[idx]['label']
        if self.transform != None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            # Convert PIL images to tensors manually if transform is False
            img1 = transforms.ToTensor()(img1)
            img2 = transforms.ToTensor()(img2)
        return img1, img2, label
    
class CustomImageTrainDataset(Dataset):
    def __init__(self, root_folder,transform):
        self.root_folder = root_folder
        self.image_list = os.listdir(root_folder)
        self.transform = transform
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img1_path = os.path.join(self.root_folder, self.image_list[idx])
        img1 = Image.open(img1_path).convert('RGB')
        img1_aug = self.transform(img1)
        img1 = transforms.ToTensor()(img1)
        return img1, img1_aug