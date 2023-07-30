import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
#dataset for running testing
class CustomImagePairDataset(Dataset):
    def __init__(self, root_folder,label_dir):
        self.root_folder = root_folder
        self.image_list = os.listdir(root_folder)
        self.normalize = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]    # Standard deviation values for normalization
        ),
        ])
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
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, label , image_1, image_2
    
#dataset for running training
class SimCLRDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = os.listdir(root_dir)
        self.normalize = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to (224, 224)
        transforms.ToTensor(),          # Convert the image to a PyTorch tensor
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean values for normalization
        std=[0.229, 0.224, 0.225]    # Standard deviation values for normalization
        ),
        ])
        self.augmented = transforms.Compose([
            # Randomly resize and crop the image
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.root_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        augmented_image = self.augmented(image)
        original_image = self.normalize(image)
        # plt.figure()
        # plt.imshow(augmented_image.permute(1, 2, 0))
        # plt.title("Augmented Image")
        # plt.axis('off')
        # plt.show()

        # # Display original image
        # plt.figure()
        # plt.imshow(original_image.permute(1, 2, 0))
        # plt.title("Original Image")
        # plt.axis('off')
        # plt.show()
        return augmented_image,original_image, image_name
    

def get_dataloader(folder_path,label_dir,batch_size,type):
    if type == "train":
        dataset = SimCLRDataset(folder_path)
    elif type == "test":
        dataset = CustomImagePairDataset(folder_path,label_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


