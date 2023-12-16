import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class BaseModel(nn.Module):
    """
    기본적인 컨볼루션 신경망 모델
    """

    def __init__(self, num_classes):
        """
        모델의 레이어 초기화

        Args:
            num_classes (int): 출력 레이어의 뉴런 수
        """
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 입력 이미지 텐서

        Returns:
            x (torch.Tensor): num_classes 크기의 출력 텐서
        """
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()

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

class MyModel2(nn.Module):
    def __init__(self, num_classes_mask, num_classes_gender, num_classes_age):
        super(MyModel2, self).__init__()

        # 사전 훈련된 ResNet50 모델 불러오기
        self.resnet = models.resnet50(pretrained=True)
        # ResNet50의 마지막 완전연결층 제거
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # 각 태스크에 대한 추가적인 커스텀 레이어
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2048, 512)
        self.fc_mask = nn.Linear(512, num_classes_mask)
        self.fc_gender = nn.Linear(512, num_classes_gender)
        self.fc_age = nn.Linear(512, num_classes_age)

    def forward(self, x):
        # ResNet50 특징 추출
        x = self.resnet(x)
        x = x.view(x.size(0), -1)  # 평탄화(Flatten)
        
        # 커스텀 레이어 적용
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        mask_outs = self.fc_mask(x)
        gender_outs = self.fc_gender(x)
        age_outs = self.fc_age(x)

        return torch.cat((mask_outs, gender_outs, age_outs), dim=1)