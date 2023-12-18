import torch.nn as nn
import torch.nn.functional as F
from torch import nn
import timm

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