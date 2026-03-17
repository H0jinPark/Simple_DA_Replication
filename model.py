import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=31):
    # 1. 깃털처럼 가벼운 MobileNetV2 등판!
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # 2. 뇌 이식 수술 (주의: ResNet이랑 구조가 살짝 다릅니다!)
    # ResNet은 마지막 층 이름이 'fc'였지만, 
    # MobileNetV2는 'classifier'라는 묶음의 [1]번째에 입이 달려있어요.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    my_model = get_model()
    print("MobileNetV2 이식 완료! 깃털처럼 가볍습니다 ㅋㅋㅋ")