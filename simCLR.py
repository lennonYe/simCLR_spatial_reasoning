import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torchvision.models.resnet import ResNet, resnet50,resnet18
from custom_dataset import get_dataloader
import matplotlib.pyplot as plt
import os
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader
from info_nce import InfoNCE

def visual_iou(thresholds, ious_list, auc):
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.subplots_adjust(left=0.3)
    plt.title('IOU vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('IOU')
    plt.plot(thresholds, ious_list, '-o', color='red', linewidth=2, label='IOU')
    plt.fill_between(thresholds, ious_list, color='blue', alpha=0.1)
    plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)
    plot_name = "simCLR.png"
    file_path = os.path.join('vis', plot_name)  
    if not os.path.exists('vis'):
        os.mkdir('vis')
    file_path = os.path.join('vis', plot_name)  
    plt.savefig(file_path)
    plt.close()


def contrastive_loss(z1, z2, temperature=0.75):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = z1.shape[0]
    embedding_size = 128
    query = z1
    positive_key = z2
    negative_keys = []
    num_negatives = 2 * (batch_size - 1)  # Number of negative examples for each query image
    negative_keys = torch.empty(batch_size, num_negatives, embedding_size, device=device)
    for i in range(batch_size):
        index = 0
        for j in range(batch_size):
            if i != j:
                # Concatenate the negative keys for each query image
                # print("Shape of z1",z1[j].shape)
                # print("Shape of z2 is:",z2[j].shape)
                # Concatenate the negative keys for each query image
                negative_keys[i,index] = z1[j]
                index += 1
                negative_keys[i,index] = z2[j]
                index += 1 
    # print("positive key is:",positive_key)
    # print("negative key is:",negative_keys)
    # print("query is",query)

    loss = InfoNCE(temperature=temperature, reduction='mean', negative_mode='paired')
    output = loss(query, positive_key, negative_keys)
    print(output)
    return output
    

class SimCLRModel(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLRModel, self).__init__()
        self.base_encoder = base_encoder
        num_features = self.get_num_features()
        self.projection_head = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.BatchNorm1d(num_features),
            nn.ReLU(),
            nn.Linear(num_features, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
    def get_num_features(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            num_features = self.base_encoder(dummy_input).view(1, -1).shape[1]
        return num_features
    def forward(self, img1, img2):
        # Encode both images using the base encoder
        z1 = self.base_encoder(img1)
        z2 = self.base_encoder(img2)
        # Flatten the encoded features
        z1 = z1.view(z1.size(0), -1)
        z2 = z2.view(z2.size(0), -1)
        # Compute the projections for contrastive loss
        proj_z1 = self.projection_head(z1)
        proj_z2 = self.projection_head(z2)

        return proj_z1, proj_z2

# Load the base encoder
base_encoder = resnet18(pretrained=True) 
# Modify the base encoder to remove the final classification layer
base_encoder = nn.Sequential(*list(base_encoder.children())[:-1])
# Create the SimCLR model with the modified base encoder
simclr_model = SimCLRModel(base_encoder)
training_folder_path = "datasets/realworld/training/images"
testing_folder_path = "datasets/realworld/testing/images"
train_label_folder = "datasets/realworld/training/training_csv.csv"
test_label_folder = "datasets/realworld/testing/testing_csv.csv"
batch_size = 16
# train_loader = get_dataloader(training_folder_path,train_label_folder, batch_size,True,"test")
test_loader = get_dataloader(testing_folder_path,test_label_folder, batch_size,"test")
train_loader = get_dataloader(training_folder_path,train_label_folder,batch_size,"train")
epochs = 10
optimizer = torch.optim.Adam(simclr_model.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*epochs)

# Training loop
for epoch in range(epochs):
    simclr_model.train()  # Set the model to training mode
    total_loss = 0.0
    for batch in train_loader:
        augmented_image,original_image = batch
        proj_z1, proj_z2 = simclr_model(original_image,augmented_image)
        # Compute the contrastive loss between the projections
        loss = contrastive_loss(proj_z1, proj_z2)
        total_loss += loss.item()
        # Backpropagate and update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Avg. Loss: {avg_loss:.4f}")


# Testing
simclr_model.eval()  # Set the model to evaluation mode
total_loss = 0.0
total_batches = len(test_loader)
thresholds = [i / 100 for i in range(0, 101, 25)]
ious_list = []
true_positives = 0
false_positives = 0
false_negatives = 0
max_similarity = 1.0
# with torch.no_grad():  # No need to calculate gradients during testing
#     for batch_idx,batch in enumerate(test_loader):
#         img1,img2,label,image_1,image_2= batch
#         proj_z1, proj_z2 = simclr_model(img1, img2)

#         similarities = torch.matmul(F.normalize(proj_z1, dim=1), F.normalize(proj_z2, dim=1).t())
#         max_similarity = max(max_similarity, similarities.max().item())
for threshold_percent in thresholds:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    threshold = float(max_similarity * threshold_percent)
    print("Current threshold is:",threshold)
    with torch.no_grad():  # No need to calculate gradients during testing
        for batch_idx, batch in enumerate(test_loader):
            img1,img2,label,image_1,image_2= batch
            proj_z1, proj_z2 = simclr_model(img1, img2)
            proj_z1_normalized = F.normalize(proj_z1, dim=1)
            proj_z2_normalized = F.normalize(proj_z2, dim=1)
            similarities = F.cosine_similarity(proj_z1_normalized.unsqueeze(1), proj_z2_normalized.unsqueeze(0), dim=2)
            binary_predictions = (similarities >= threshold).float()
            print("Current image1 is:",image_1)
            print("Current image2 is:",image_2)
            print("Current prediction is:",binary_predictions)
            print("Current label is:",label)
            print("similarity is:",similarities)
            true_positives += ((binary_predictions == 1) & (label == 1)).sum().item()
            false_positives += ((binary_predictions == 1) & (label == 0)).sum().item()
            false_negatives += ((binary_predictions == 0) & (label == 1)).sum().item()
    iou = true_positives / (true_positives + false_positives + false_negatives)
    ious_list.append(iou)

auc = np.trapz(ious_list, thresholds)

# Visualize IOU
visual_iou(thresholds, ious_list, auc)
avg_loss = total_loss / total_batches
print(f"Average Loss on Test Set: {avg_loss:.4f}")
iou = true_positives / (true_positives + false_positives + false_negatives)
print(f"Intersection over Union (IOU): {iou:.4f}")