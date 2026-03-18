import torch
import torch.nn as nn
from torchvision import models

class MMD_MobileNetV3(nn.Module):
    def __init__(self, num_classes=31):
        super(MMD_MobileNetV3, self).__init__()
        # 1. 뼈대(Backbone) 불러오기
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # 2. 이미지에서 기본 특징을 뽑아내는 부분
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        
        # 3. 분류기(classifier) 해체 및 재조립!
        # MobileNetV3_Small의 classifier 구조: [0]Linear, [1]Hardswish, [2]Dropout, [3]Linear
        # 최종 [3]번 층을 통과하기 전까지의 값(0~2번 통과)을 '특징 벡터(Feature)'로 쓸 거야.
        self.feature_extractor = nn.Sequential(
            backbone.classifier[0],
            backbone.classifier[1],
            backbone.classifier[2]
        )
        
        # 4. 최종 정답을 맞추는 출력층 (클래스 개수 31개에 맞게 갈아끼우기)
        in_features = backbone.classifier[3].in_features
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 뼈대 통과
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # 🌟 도메인 거리를 잴 때 사용할 '특징 벡터 (Feature)'
        f = self.feature_extractor(x)
        
        # 🌟 최종 정답을 예측할 '출력값 (Output)'
        out = self.fc(f)
        
        # MMD 학습을 위해 사이좋게 두 개를 다 던져줍니다!
        return f, out

# train_MMD.py에서 이 함수를 부를 거예요!
def get_model(num_classes=31):
    return MMD_MobileNetV3(num_classes=num_classes)