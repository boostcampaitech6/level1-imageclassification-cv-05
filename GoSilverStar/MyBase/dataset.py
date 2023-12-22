from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import cv2
from facenet_pytorch import MTCNN


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

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN(keep_all=True, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

    def detect_and_crop(self, image):
        # Detect faces
        boxes, _ = self.detector.detect(image)

        # If a face is detected, crop the image around the first face
        if boxes is not None and len(boxes) > 0:
            x, y, w, h = boxes[0]
            cropped_image = image.crop((x, y, w, h))
            return cropped_image

        # If no face is detected, return None
        return None

class BasicAugmentation:
    def __init__(self, resize):
        self.face_detector = FaceDetector()
        self.transform = A.Compose(
            [   
                # Replace or add the face detection/cropping step here
                A.Lambda(image=self.apply_face_detection),
                A.Resize(resize[0], resize[1]),
                A.Normalize(),
                ToTensorV2()
            ]
        )
    def apply_face_detection(self, image,**kwargs):
        # Convert OpenCV image to PIL image for MTCNN
        image_pil = Image.fromarray(image)

        # Apply face detection and cropping
        cropped_image = self.face_detector.detect_and_crop(image_pil)

        # If a face is detected
        if cropped_image is not None:
            return np.array(cropped_image)  # Convert back to numpy array

        # If no face is detected, apply center crop
        return A.CenterCrop(320, 256)(image=image)['image']
    
    def __call__(self, image):
        return self.transform(image=image)
    
class CustomAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""
    
    def __init__(self, resize):
        self.face_detector = FaceDetector()
        self.transform = A.Compose(
            [
                # A.CenterCrop(height=300, width=300),
                A.Lambda(image=self.apply_face_detection),
                A.Resize(resize[0],resize[1]), # Image.BILINEAR is `interpolation=1` in Albumentations
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                A.Rotate(limit=10),
                A.HorizontalFlip(p=1),  # p=1 means the flip is applied to all images
                A.Normalize(),
                ToTensorV2()
            ]
        )
    def apply_face_detection(self, image,**kwargs):
        # Convert OpenCV image to PIL image for MTCNN
        image_pil = Image.fromarray(image)

        # Apply face detection and cropping
        cropped_image = self.face_detector.detect_and_crop(image_pil)

        # If a face is detected
        if cropped_image is not None:
            return np.array(cropped_image)  # Convert back to numpy array

        # If no face is detected, apply center crop
        return A.CenterCrop(320, 256)(image=image)['image']
    def __call__(self, image):
        return self.transform(image = image)

    
class CustomAugmentation2:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize):
        self.face_detector = FaceDetector()
        self.transform = A.Compose(
            [
                # A.CenterCrop(height=300, width=300),
                A.Lambda(image=self.apply_face_detection),
                A.Resize(resize[0],resize[1]), # Image.BILINEAR is `interpolation=1` in Albumentations
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                A.RandomContrast(limit=0.2, p=0.5),
                A.Normalize(),
                ToTensorV2()
            ]
        )
    def apply_face_detection(self, image,**kwargs):
        # Convert OpenCV image to PIL image for MTCNN
        image_pil = Image.fromarray(image)

        # Apply face detection and cropping
        cropped_image = self.face_detector.detect_and_crop(image_pil)

        # If a face is detected
        if cropped_image is not None:
            return np.array(cropped_image)  # Convert back to numpy array

        # If no face is detected, apply center crop
        return A.CenterCrop(320, 256)(image=image)['image']
    def __call__(self, image):
        return self.transform(image = image)
    
# class FaceCrop:
#     """얼굴을 크롭하는 변환 클래스"""

#     def __init__(self, fallback_crop_size=(320, 256)):
#         self.fallback_crop = A.CenterCrop(*fallback_crop_size, always_apply=True)

#     def __call__(self, image):
#         # RetinaFace를 사용하여 얼굴 감지
#         faces = RetinaFace.detect_faces(image)
#         if len(faces) == 0:
#             # 얼굴이 감지되지 않으면 중앙에서 크롭
#             return self.fallback_crop(image=image)['image']

#         # 첫 번째 감지된 얼굴의 영역 가져오기
#         face_coordinates = list(faces.values())[0]['facial_area']
#         x, y, w, h = face_coordinates

#         # 얼굴 영역 크롭
#         face_crop = image[y:h, x:w]
#         return face_crop    
    
# class FaceAugmentation:
#     """커스텀 Augmentation을 담당하는 클래스"""

#     def __init__(self, resize):
#         self.transform = A.Compose(
#             [   FaceCrop(),
#                 A.Resize(resize[0],resize[1]),
#                 A.Normalize(),
#                 ToTensorV2()
#             ]
#         )

#     def __call__(self, image):
#         return self.transform(image= image)