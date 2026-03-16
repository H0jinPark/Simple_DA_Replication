import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=31):
    # 1. 이미 학습된 ResNet50 모델 불러오기 (Pre-trained)
    # weights=models.ResNet50_Weights.DEFAULT 가 요즘 최신 방식이에요!
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # 2. 마지막 층(Fully Connected)을 우리 데이터셋에 맞게 바꾸기
    # ResNet50의 마지막 출력은 보통 1000개인데, Office-31은 클래스가 31개니까요!
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    return model

if __name__ == "__main__":
    # 모델이 잘 만들어졌는지 테스트!
    my_model = get_model()
    print(my_model) # 모델 구조가 주르륵 뜰 거예요
    print("\n모델 생성 성공!")