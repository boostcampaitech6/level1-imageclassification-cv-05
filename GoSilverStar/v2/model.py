import torch.nn as nn
import torch.nn.functional as F
import timm

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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm="bnorm", relu=True):
        super().__init__()

        layers = []
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             bias=bias)]

        if norm == "bnorm":
            layers += [nn.BatchNorm2d(num_features=out_channels)]

        if relu:
            layers += [nn.ReLU()]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bias=True, norm="bnorm", short_cut=False, relu=True, init_block=False):
        super().__init__()

        layers = []


        if init_block:
          init_stride = 2
        else:
          init_stride = stride

        # 1st conv
        layers += [ConvBlock(in_channels=in_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=init_stride, padding=padding,
                         bias=bias, norm=norm, relu=relu)]

        # 2nd conv
        layers += [ConvBlock(in_channels=out_channels, out_channels=out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         bias=bias, norm=norm, relu=False)]

        self.resblk = nn.Sequential(*layers)


        self.short_cut = nn.Conv2d(in_channels, out_channels, (1,1), stride=2)

    def forward(self, x, short_cut=False):
        if short_cut:
            return self.short_cut(x) + self.resblk(x)
        else:
            return x + self.resblk(x) # residual connection
        
class ResNet(nn.Module):
    def __init__(self, num_classes):
        in_channels = 3
        out_channels = num_classes
        nker=64
        norm="bnorm"
        nblk=[3,4,6,3]
        
        super(ResNet, self).__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=1, bias=True, norm=None, relu=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        res_1 = ResBlock(nker, nker, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
        self.res_1 = nn.Sequential(*[res_1 for _ in range(nblk[0])])

        res_2 = ResBlock(nker*2, nker*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
        self.res_2_up = ResBlock(nker, nker*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True, init_block=True)
        self.res_2 = nn.Sequential(*[res_2 for _ in range(nblk[1]-1)])

        res_3 = ResBlock(nker*2*2, nker*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
        self.res_3_up = ResBlock(nker*2, nker*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True, init_block=True)
        self.res_3 = nn.Sequential(*[res_3 for _ in range(nblk[2]-1)])

        res_4 = ResBlock(nker*2*2*2, nker*2*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True, init_block=True)
        self.res_4_up = ResBlock(nker*2*2, nker*2*2*2, kernel_size=3, stride=1, padding=1, bias=True, norm=norm, relu=True)
        self.res_4 = nn.Sequential(*[res_4 for _ in range(nblk[3]-1)])

        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(nker*2*2*2, 18)

    def forward(self, x):
        x = self.enc(x)
        x = self.max_pool(x)
        x = self.res_1(x)
        x = self.max_pool(x)

        x = self.res_2_up(x, short_cut=True)
        x = self.res_2(x)
        x = self.max_pool(x)

        x = self.res_3_up(x, short_cut=True)
        x = self.res_3(x)
        x = self.max_pool(x)

        x = self.res_4_up(x, short_cut=True)
        x = self.res_4(x)

        x = self.avg_pooling(x)
        x = x.view(x.shape[0], -1)
        out = self.fc(x)
        return out

# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.features = timm.create_model('efficientnet_b0', pretrained=True)
        
        in_features = self.features.classifier.in_features
        self.features.classifier = nn.Linear(in_features, num_classes)

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
        x = self.features(x)
        return x
