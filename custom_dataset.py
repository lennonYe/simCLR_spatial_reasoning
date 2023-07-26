import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
class CustomImagePairDataset(Dataset):
    def __init__(self, root_folder,label_dir,transform=None):
        self.root_folder = root_folder
        self.image_list = os.listdir(root_folder)
        self.transform = transform
        self.labels = pd.read_csv(label_dir, names=['image_1', 'image_2', 'label'])
    def __len__(self):
        return len(self.labels)
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
        return img1, img2, label , image_1, image_2
    
# class CustomImageTrainDataset(Dataset):
#     def __init__(self, root_folder,transform):
#         self.root_folder = root_folder
#         self.image_list = os.listdir(root_folder)
#         self.transform = transform
#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, idx):
#         img1_path = os.path.join(self.root_folder, self.image_list[idx])
#         img1 = Image.open(img1_path).convert('RGB')
#         img1_aug = self.transform(img1)
#         img1 = transforms.ToTensor()(img1)
#         return img1, img1_aug

class SimCLRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)
        self.transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]    # Standard deviation values for normalization
        ),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)], p=0.5)
        ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(image_name).convert("RGB")
        augmented_image = self.transform(image)
        original_image = transforms.ToTensor()(image)
        return augmented_image,original_image
    

def get_dataloader(folder_path,label_dir,batch_size,type):
    if type == "train":
        dataset = SimCLRDataset(folder_path)
    elif type == "test":
        dataset = CustomImagePairDataset(folder_path,label_dir,None)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


