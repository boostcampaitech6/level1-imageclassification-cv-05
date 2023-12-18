import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 데이터셋 정의 (예시를 위한 간단한 데이터셋)
class SimpleDataset(Dataset):
    def __init__(self):
        # 예시 데이터: 여기에서는 무작위 데이터를 생성합니다.
        # 실제 사용시에는 이미지 데이터와 레이블을 로드하는 코드로 대체해야 합니다.
        self.data = torch.randn(100, 3, 224, 224)
        self.labels = torch.randint(0, 3, (100, 3))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

# 모델 정의
class MultiLabelEfficientNet(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelEfficientNet, self).__init__()
        # EfficientNet 로드
        self.base_model = models.efficientnet_b0(pretrained=True)
        # 분류를 위한 새로운 레이어 추가
        self.fc = nn.Linear(self.base_model.classifier[1].in_features, num_classes)

    def forward(self, x):
        x = self.base_model(x)
        x = self.fc(x)
        return x

# 하이퍼파라미터 설정
num_classes = 3  # 클래스의 수
learning_rate = 0.001
batch_size = 16
num_epochs = 10

# 데이터셋 및 데이터 로더 준비
dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 초기화
model = MultiLabelEfficientNet(num_classes)

# 손실 함수 및 최적화 알고리즘 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    for images, labels in dataloader:
        # 예측 및 손실 계산
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
