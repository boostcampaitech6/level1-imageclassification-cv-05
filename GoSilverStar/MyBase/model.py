import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import timm

class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.model = timm.create_model('efficientnet_b3', pretrained=True) #features ->model로 
        
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        x = self.model(x)
        return x