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
        age = torch.tensor(int(self.dataframe.iloc[idx]['age']), dtype=torch.int)
        
        # Return the image and the corresponding labels
        return image, mask_label, gender_label, age_label, total_label, age
    

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
    
class CustomAugmentation2:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize):
        self.transform = A.Compose(
            [   
                A.Resize(resize[0], resize[1]),  # 이미지 크기 조정
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7), p=0.5),  # 가우시안 블러
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.5),  # 그리드 왜곡
                    A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=0.5)  # 컷아웃
                ], p=1),
                A.Normalize(),  # 정규화
                ToTensorV2()  # 텐서 변환
            ]
        )

    def __call__(self, image):
        return self.transform(image= image)
    
########################################################################

class None_aug:    # 원본 이미지 반환
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.Normalize(),
            ToTensorV2()
            ])

    def __call__(self, image):
        return self.transform(image= image)

class Horizontal_Rotate_aug:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.HorizontalFlip(p=1),
            A.Rotate(limit=20),
            A.Normalize(),
            ToTensorV2()
            ])
    def __call__(self, image):
        return self.transform(image= image)
    
class Rotate_aug:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.Rotate(20),
            A.HorizontalFlip(p=0.1),
            ToTensorV2()
            ])
    def __call__(self, image):
        return self.transform(image= image)
    
class ColorJitter_Flip_aug:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            #A.ColorJitter(0.5, 0.5, 0.5, 0.5),
            A.HorizontalFlip(p=0.5),
            ToTensorV2()
            ])
    def __call__(self, image):
        return self.transform(image= image)
    
class ColorJitter_aug:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.ColorJitter(0.5, 0.5, 0.5, 0.5),
            A.HorizontalFlip(p=0.1),
            ToTensorV2()
        ])
    def __call__(self, image):
        return self.transform(image= image)
    
class ColorJitter_aug_for_male:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            #A.ColorJitter(0.3, 0, 0.3, 0),
            A.Normalize(),
            ToTensorV2()
        ])
    def __call__(self, image):
        return self.transform(image= image)['image']
    
class ColorJitter_aug_for_female:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.ToGray(p=0.1),
            ToTensorV2()
        ])
    def __call__(self, image):
        return self.transform(image= image)
    
class Grayscale_aug:
    def __init__(self, resize):
        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.ToGray(p=1),
            A.HorizontalFlip(p=0.1),
            ToTensorV2()
        ])
    def __call__(self, image):
        return self.transform(image= image)  

class Sharpness_augmix:
    def __init__(self, resize):
        augs = [A.HorizontalFlip(always_apply=True),
        A.Blur(always_apply=True),
        A.Cutout(always_apply=True),
        A.IAAPiecewiseAffine(always_apply=True)]

        self.transform = A.Compose([
            A.Resize(resize[0],resize[1]),
            A.Sharpen(alpha=(0.2,0.5),lightness=(0.5,1),always_apply=True),
            A.HorizontalFlip(p=0.1),
            ToTensorV2()
        ])
    def __call__(self, image):
        return self.transform(image= image)
############################################################