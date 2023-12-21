import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import timm
import torchvision.models as models

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """
        # self.model = timm.create_model(model_name='mobilenetv3_small_050', pretrained=True)
        self.model = timm.create_model(model_name= 'swinv2_base_window8_256', pretrained=True)
        self.model.head.fc = nn.LazyLinear(num_classes)
        # self.model.classifier = nn.LazyLinear(num_classes)
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # # classifier 의 파라미터는 훈련을 통해 업데이트되도록 설정
        # for param in self.model.head.fc.parameters():
        #     param.requires_grad = True

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x
    
    
class MultiLabelModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # self.model = timm.create_model(model_name='mobilenetv3_small_050', pretrained=True)
        self.model = timm.create_model(model_name= 'swinv2_base_window8_256', pretrained=True)
        # self.model.head.fc = nn.LazyLinear(num_classes)
        
        # self.mask_head = self.model.head
        self.mask_head = nn.LazyLinear(3)
        
        # self.gender_head = self.model.head
        self.gender_head = nn.LazyLinear(2)
        
        # self.age_head = self.model.head
        self.age_head = nn.LazyLinear(3)
        
        # del self.model.head
        
        
        # for param in self.model.parameters():
        #     param.requires_grad = False

        # classifier 의 파라미터는 훈련을 통해 업데이트되도록 설정

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        
        mask = self.mask_head(x)
        gender = self.gender_head(x)
        age = self.age_head(x)
        
        return mask, gender, age
    
# m = MultiLabelModel(18)
# print(m)
    
class MyResnet50(nn.Module):
    def __init__(self, num_classes):
        super(MyResnet50, self).__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

        # 사전 훈련된 ResNet50 모델 불러오기
        self.resnet = models.resnet50(pretrained=True)
        
        # ResNet50의 마지막 완전연결층 제거
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # 추가적인 커스텀 레이어
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
         # ResNet50 특징 추출
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # 평탄화(Flatten)
        
        # 커스텀 레이어 적용
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

class MyResNeXt(nn.Module):
    def __init__(self, num_classes):
        super(MyResNeXt, self).__init__()
        # 사전 훈련된 ResNeXt 모델 불러오기
        self.resnext = models.resnext50_32x4d(pretrained=True)
        # ResNeXt의 마지막 완전연결층 제거
        self.resnext.fc = nn.Linear(self.resnext.fc.in_features, num_classes)

    def forward(self, x):
        # ResNeXt 특징 추출 및 분류
        x = self.resnext(x)
        return x