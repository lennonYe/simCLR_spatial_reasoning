import os
import torch
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import WandbLogger
#from pytorch_lightning.plugins import DDPPlugin
from omegaconf import OmegaConf
from sklearn.model_selection import train_test_split

from NewLoader import AVDDataset, Subset
from visLoader import VisAVDDataset
import matplotlib.pyplot as plt
from lightning import ClassificationModel

import gc
torch.cuda.empty_cache()
gc.collect()

os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

#parser = argparse.ArgumentParser()
#parser.add_argument('--conf', type=str, help="path config files")

#args = parser.parse_args()

def visualizeAUC(
    auc, 
    iou, 
    config_filename, 
    backbone, 
    balanced, 
    attention,
    dataset_type,
    ):
    """
    Plot the IOU on a graph with the AUC
    """
    x = np.arange(0, 1.05, 0.05)

    # Set x and y axes limits
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.subplots_adjust(left=0.3)

    # Set plot title and axes labels
    plt.title('AUC')
    plt.xlabel('Thresholds')
    plt.ylabel('IOU')

    # Draw the plot
    plt.plot(x, iou, '-o', color='red', linewidth=2, label='IOU')

    # Fill in the area under the curve
    plt.fill_between(x, iou, color='blue', alpha=0.1)

    # Label the AUC
    plt.text(0.5, 0.5, 'AUC = %0.2f' % auc)

    # Write config
    plt.figtext(0, 0.55, 'Backbone: %s' % backbone, fontsize=8)
    plt.figtext(0, 0.5, 'Class Balanced: %s' % balanced, fontsize=8)
    plt.figtext(0, 0.45, 'With Attention: %s' % attention, fontsize=8)

    # Save the plot
    full_path = os.path.join('vis', dataset_type+'_'+config_filename)
    plt.savefig(full_path)

    plt.close()

def calculateIOU(
    predictions,
    labels,
    threshold=0.5, 
    ):
    """
    Calculate IOU using the predictions and labels
    """
    # Get the predictions above the threshold
    pred = predictions > threshold

    # Calculate the intersection and union
    intersection = np.logical_and(pred, labels)
    union = np.logical_or(pred, labels)
    intersection = np.sum(intersection)
    union = np.sum(union)

    if union == 0:
        return 0.0
    return intersection / union

def getPredictions(
    model, 
    dataloader,
    csv_filename,
    class_balanced=False,
    concat_csr=False,
    ):
    """
    Get the predictions from the model
    """
    model.eval()
    csv_output = []
    pred = np.array([])
    labels = np.array([])
    with torch.no_grad():
        for i, (filenames, img1, img2, label, csr1, csr2) in enumerate(tqdm(dataloader)):
            img1 = img1.cuda()
            img2 = img2.cuda()

            # Get the predictions
            if concat_csr:
                csr1 = csr1.cuda()
                csr2 = csr2.cuda()
                out = model(img1, img2, csr1, csr2)[:, 1].cpu().numpy()
                pred = np.append(pred, out)
            else:
                out = model(img1, img2)[:, 1].cpu().numpy()
                pred = np.append(pred, out)
            
            labels = np.append(labels, label)

            # Get entry for csv
            filename1 = filenames[0][0]
            filename2 = filenames[1][0]

            entry = [filename1, filename2, out.item(), label.item()]

            csv_output.append(entry)
    
    # Save the csv
    df = pd.DataFrame(csv_output)
    df.to_csv('vis/'+csv_filename+'.csv', index=False, header=False)

    return pred, labels

def calculateAUC(
    model, 
    dataloader,
    csv_filename,
    class_balanced=False,
    concat_csr=False,
    ):
    """
    Calculate the area under the curve using the IOUs calculated from each threshold from 0 to 1 with 0.05 increments
    """
    ious = []
    thresholds = np.arange(0, 1.05, 0.05)
    pred, labels = getPredictions(model, dataloader, csv_filename, class_balanced=class_balanced, concat_csr=concat_csr)

    for t in thresholds:
        iou = calculateIOU(pred, labels, threshold=t)
        ious.append(iou)
    
    print(f'IOUs: {ious}')
    auc = np.trapz(ious, thresholds)
    print(f'AUC: {auc}')
    return auc, ious

def ablation(data_dir, config_dir):
    # Create vis directory
    if not os.path.exists('vis'):
        os.makedirs('vis')

    # Clean Log files
    if os.path.exists('lightning_logs'):
        for file in os.listdir('lightning_logs'):
            if file.startswith('version'):
                shutil.rmtree(os.path.join('lightning_logs', file))

    # Print total number of configs
    print(f'Total number of configs: {len(os.listdir(config_dir))}')

    # Need to determine train/val split
    # The dataset has the following format:
    #   dataset
    #   ├── scene1
    #   │   ├── floor1
    #   │   │   ├── saved_obs
    #   │   │   │   ├── boxes
    #   │   │   │   ├── csr
    #   │   │   │   ├── resnet
    #   │   │   │   ├── vgg
    #   │   │   │   ├── vit
    #   │   │   │   ├── labels.csv
    #   │   │   │   ├── best_obs_1.png
    #   │   │   │   ├── best_obs_2.png
    #   │   │   │   ├── ...

    # Assign each Scene to either train or val
    # 80/20 split
    scenes = os.listdir(data_dir)
    train_scenes, val_scenes = train_test_split(
        scenes,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Iterate through all the config files in the directory
    for config in os.listdir(config_dir):
        print(f'Testing config: {config}')
        conf = OmegaConf.load(os.path.join(config_dir, config))
        print(OmegaConf.to_yaml(conf))

        # Prepare datasets
        dataset = AVDDataset(
            train_scenes,
            conf.backbone_str
        )

        # Split dataset into train and val
        train_idx, val_idx = train_test_split(
            np.arange(len(dataset)), 
            test_size=0.2, 
            random_state=42,
            shuffle=True,
            stratify=dataset.targets
        )

        train_dataset = Subset(dataset, train_idx)
        val_dataset = Subset(dataset, val_idx)

        print("\nStarting training")
        model = ClassificationModel(
            train_dataset,
            val_dataset,
            conf.lr,
            conf.batch_size,
            conf.backbone_str,
            conf.class_balanced,
            conf.with_attention,
            conf.concat_csr,
        )
        
        # Get config filename from path
        config_filename = config.split('/')[-1].split('.')[0]

        # Create checkpoint callback directory for each config
        dir_name = f'checkpoints/spatial-reasoning/{config_filename}'
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        # If checkpoints exist, delete them
        for file in os.listdir(dir_name):
            os.remove(os.path.join(dir_name, file)) 

        checkpoint_callback = ModelCheckpoint(
            dirpath=dir_name,
            filename=f'{config_filename}' + '-{epoch:02d}-{val_loss:.2f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1,
        )

        trainer = Trainer(max_epochs=20, gpus=1, callbacks=[checkpoint_callback])
        #trainer = Trainer(fast_dev_run=True, gpus=1)
        print("\nFitting")
        trainer.fit(model)

        # Use the callback to get the best checkpoint
        best_model_path = checkpoint_callback.best_model_path
        print(f"best_model_path: {best_model_path}")

        # Load the best checkpoint
        best_model = ClassificationModel.load_from_checkpoint(best_model_path).to('cuda')

        """
        Visualization stuff
        """

        dataset = VisAVDDataset(
            train_scenes,
            conf.backbone_str
        )

        test_dataset = VisAVDDataset(
            val_scenes,
            conf.backbone_str
        )

        train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8)

        # Calculate AUC
        print("\nCalculating AUC for train")
        train_auc, train_iou = calculateAUC(best_model, train_loader, 'train_'+config_filename, conf.class_balanced, conf.concat_csr)
        print("\nCalculating AUC for val")
        test_auc, test_iou = calculateAUC(best_model, test_loader, 'test_'+config_filename, conf.class_balanced, conf.concat_csr)

        # Visualize AUC
        if not os.path.exists('vis'):
            os.makedirs('vis')
        visualizeAUC(
            train_auc,
            train_iou, 
            config_filename, 
            conf.backbone_str,
            conf.class_balanced,
            conf.with_attention,
            'train'
        )
        visualizeAUC(
            test_auc,
            test_iou, 
            config_filename, 
            conf.backbone_str,
            conf.class_balanced,
            conf.with_attention,
            'test'
        )

        # Clean up memory for next config
        del model
        del best_model
        del dataset
        del test_dataset
        del train_loader
        del test_loader
        del train_dataset
        del val_dataset
        torch.cuda.empty_cache()
        gc.collect()
        print(f'Finished testing config: {config}\n\n')
    
    print('Finished testing all configs!')