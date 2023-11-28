import torch
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as F
from PIL import Image
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import config
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def collate_fn(batch):
        return tuple(zip(*batch))


class Flickr27Dataset(Dataset):
    def __init__(self, root_folder, annotation_file, class_name_to_label ,transform=None):
        self.root_folder = root_folder
        self.annotations = annotation_file
        self.transform = transform
        self.ClasstoLabel = class_name_to_label
        
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_name = self.annotations.iloc[idx][0]
        image_path = f"{self.root_folder}/{image_name}"

        # Load image
        image = Image.open(image_path).convert("RGB")
        wt, ht = image.size
        image = image.resize((config.WIDTH, config.HEIGHT))
        image = np.array(image).astype(np.float32)
        image /= 255.0
        
        
        # Extract bounding box coordinates and class label
        xmin = self.annotations.iloc[idx][3]
        ymin = self.annotations.iloc[idx][4]
        xmax = self.annotations.iloc[idx][5]
        ymax = self.annotations.iloc[idx][6]
        
        # change bounding box as per resize
        
        xmin_corr = (xmin/wt)*config.WIDTH
        xmax_corr = (xmax/wt)*config.WIDTH
        ymin_corr = (ymin/ht)*config.HEIGHT
        ymax_corr = (ymax/ht)*config.HEIGHT
        
        
        box = [xmin_corr, ymin_corr, xmax_corr, ymax_corr]
        box = [float(coord) for coord in box]
        class_name = self.annotations.iloc[idx][1]
        class_label = self.ClasstoLabel[class_name]
         
        
        # Create target dictionary containing bounding box and class information
        target = {
            'image_id' : torch.tensor([idx]),
            'boxes' : torch.tensor([box], dtype=torch.float32),
            'labels' : torch.tensor([class_label], dtype=torch.int64),
        }
        
        if self.transform:
            sample = {
                'image': image,
                'bboxes': target['boxes'],
                'labels': target['labels']
            }
            sample = self.transform(**sample)
            image = sample['image']
            
            target['boxes'] = torch.as_tensor(sample['bboxes'])
 

        return image, target



class Flickr27DataModule(pl.LightningDataModule):
    def __init__(self, root_folder, annotation_file, batch_size=2, val_split=0.1):
        super(Flickr27DataModule, self).__init__()
        self.root_folder = root_folder
        self.annotation_file = annotation_file
        self.batch_size = batch_size
        self.val_split = val_split
        
    

    def setup(self, stage=None):
        # Read the annotation file into a DataFrame
        
        columns = ["imgname", "classname", "class", "xmin", "ymin", "xmax", "ymax"]
        annotations = pd.read_csv(self.annotation_file, delimiter=' ', header=None, names=columns, index_col=False)

        class_names = annotations['classname'].unique()
        
        class_name_to_label = {class_name: label for label, class_name in enumerate(class_names)}
        
        # Split the dataset into training and validation sets
        train_data, val_data = train_test_split(annotations, test_size=self.val_split, random_state=42)

      
        self.train_dataset = Flickr27Dataset(
            root_folder=self.root_folder,
            annotation_file=train_data,
            class_name_to_label=class_name_to_label,
            transform=get_train_transform())

        self.val_dataset = Flickr27Dataset(
            root_folder=self.root_folder,
            annotation_file=val_data,
            class_name_to_label=class_name_to_label,
            transform=get_valid_transform())
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,collate_fn=collate_fn,shuffle=True,num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,collate_fn=collate_fn,shuffle=False,num_workers=15)

    def test_dataloader(self):
        # For simplicity, using the same DataLoader for testing as for validation
        return DataLoader(self.val_dataset, batch_size=self.batch_size,collate_fn=collate_fn,shuffle=False,num_workers=15)
