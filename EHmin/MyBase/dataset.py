from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

class CustomDataset(Dataset):
    
    def __init__(self, dataframe, transform=None, test= False):
        """
        Custom dataset that accepts a DataFrame, a transformation function, and returns images and labels.
        
        :param dataframe: pandas DataFrame containing the image paths and labels.
        :param transform: albumentations transformation pipeline.
        """
        self.dataframe = dataframe
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.dataframe)
    
    def set_transform(self, transform):
        """변환(transform)을 설정하는 메서드"""
        self.transform = transform

    def __getitem__(self, idx):
        # Retrieve image path from dataframe
        img_path = self.dataframe.iloc[idx]['Image_path']
        
        # Load image using PIL
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image=np.array(image))['image']  # Convert to numpy array and apply transform
            
        if self.test:
            return image
        
        # Get labels from the dataframe
        mask_label = torch.tensor(self.dataframe.iloc[idx]['Mask_label'], dtype=torch.long)
        gender_label = torch.tensor(self.dataframe.iloc[idx]['Gender_label'], dtype=torch.long)
        age_label = torch.tensor(self.dataframe.iloc[idx]['Age_label'], dtype=torch.long)
        total_label = torch.tensor(self.dataframe.iloc[idx]['Total_label'], dtype=torch.long)
        
        # Return the image and the corresponding labels
        return image, mask_label, gender_label, age_label, total_label
    

    
#! 여러가지 transform 설정 

basic_transform = A.Compose([   
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])
    
transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

class CustomAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize):
        self.transform = A.Compose(
            [
                # A.CenterCrop(height=300, width=300),
                A.Resize(resize[0],resize[1]), # Image.BILINEAR is `interpolation=1` in Albumentations
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                A.Rotate(limit=10),
                A.HorizontalFlip(p=1),  # p=1 means the flip is applied to all images
                A.Normalize(),
                ToTensorV2()
            ]
        )

    def __call__(self, image):
        return self.transform(image = image)
    
class BasicAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize):
        self.transform = A.Compose(
            [   
                A.Resize(resize[0],resize[1]),
                A.Normalize(),
                ToTensorV2()
            ]
        )

    def __call__(self, image):
        return self.transform(image= image)