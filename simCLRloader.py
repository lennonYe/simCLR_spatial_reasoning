import os
import torchvision.transforms as T
from torch.utils.data import DataLoader
from custom_dataset import CustomImagePairDataset,CustomImageTrainDataset

def get_dataloader(folder_path,label_path, batch_size,transform_bool,type):
    transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if type == "test":
        if transform_bool == True:
            dataset = CustomImagePairDataset(folder_path,label_path, transform=transform)
        elif transform_bool == False:
            dataset = CustomImagePairDataset(folder_path,label_path,transform=None)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif type == "train":
        dataset = CustomImageTrainDataset(folder_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
