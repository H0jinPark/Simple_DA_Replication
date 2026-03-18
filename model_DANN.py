import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Function

# 🌟 1. 마법의 밸브 GRL (Gradient Reversal Layer) 구현
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x) # 앞으로 갈 때(Forward)는 특징을 그대로 통과시킴!

    @staticmethod
    def backward(ctx, grad_output):
        # 뒤로 갈 때(Backward)는 오차(Gradient)에 마이너스 알파(-alpha)를 곱해서 역전!
        output = grad_output.neg() * ctx.alpha 
        return output, None

# 🌟 2. DANN 모델 조립
class DANN_MobileNetV3(nn.Module):
    def __init__(self, num_classes=31):
        super(DANN_MobileNetV3, self).__init__()
        backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        
        # [위조지폐범: 특징 추출기]
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_extractor = nn.Sequential(
            backbone.classifier[0],
            backbone.classifier[1],
            backbone.classifier[2]
        )
        
        # [원래 업무: 정답 분류기]
        in_features = backbone.classifier[3].in_features
        self.class_classifier = nn.Linear(in_features, num_classes)
        
        # 🚨 [새로운 경찰: 도메인 판별기] - Source(0)인지 Target(1)인지 2진 분류
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 2) # 도메인이 2개니까 출력은 2!
        )

    def forward(self, input_data, alpha=None):
        # 1. 이미지 -> 특징 추출
        x = self.features(input_data)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feature = self.feature_extractor(x)
        
        # 2. 특징 -> 정답 예측
        class_output = self.class_classifier(feature)
        
        # 3. 알파(alpha) 값이 들어오면 GRL을 켜고 도메인 예측도 수행!
        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
            domain_output = self.domain_classifier(reverse_feature)
            return class_output, domain_output
            
        # 평가할 때는 도메인 예측이 필요 없으니 정답만 반환
        return class_output

def get_model(num_classes=31):
    return DANN_MobileNetV3(num_classes=num_classes)