import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=31):
    # 1. 깃털처럼 가벼운 MobileNetV3 Small 등판!
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # 2. 뇌 이식 수술 (V3는 마지막 층이 3번에 있어요!)
    # classifier[3]이 최종 Linear 층입니다.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    my_model = get_model()
    # 파라미터 개수 출력해서 얼마나 가벼운지 확인해볼까?
    total_params = sum(p.numel() for p in my_model.parameters())
    print(f"MobileNetV3 Small 이식 완료! 총 파라미터: {total_params:,}개")
    print("진짜 가볍다 ㅋㅋㅋ 이제 10에폭 순삭 가능!")