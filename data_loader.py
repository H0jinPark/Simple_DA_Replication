import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 이미지 전처리(Transform) 규칙 정하기
# 연구용으로 가장 많이 쓰이는 규격인 224x224 사이즈
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),       # 이미지 크기를 똑같이 맞추기
    transforms.ToTensor(),                # 파이썬 이미지를 파이치치 텐서(숫자)로 변경
    transforms.Normalize([0.485, 0.456, 0.406], # ImageNet 데이터셋의 평균과
                         [0.229, 0.224, 0.225]) # 표준편차로 색감 정규화 (필수!)
])

# 2. 데이터 경로 설정
data_root = r"C:\Users\akska\Research\Office-31"

# 도메인 이름들 (폴더명과 똑같이 적어야 해요)
domains = ['amazon', 'dslr', 'webcam']

def get_loader(domain_name, batch_size=32):
    # 해당 도메인의 폴더 경로 (예: Office-31/amazon)
    path = os.path.join(data_root, domain_name)
    
    # 폴더 구조를 그대로 읽어서 데이터셋으로 만듦
    dataset = datasets.ImageFolder(root=path, transform=data_transforms)
    
    # 데이터를 묶음(Batch) 단위로 나누어주는 로더 생성
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader

if __name__ == "__main__":
    # 테스트로 'amazon' 도메인 로더를 불러와 볼게요
    amazon_loader = get_loader('amazon')
    
    # 딱 한 묶음만 꺼내서 확인
    images, labels = next(iter(amazon_loader))
    
    print(f"이미지 묶음 크기: {images.shape}") # [32, 3, 224, 224]
    print(f"첫 번째 묶음의 라벨: {labels}")
    print("데이터 로딩 성공!")

    amazon_loader = get_loader('amazon')
    images, labels = next(iter(amazon_loader))
    
    # 첫 번째 이미지 한 장 꺼내기 [3, 224, 224]
    sample_img = images[0].numpy().transpose((1, 2, 0)) 
    
    # 정규화된 이미지를 다시 원래대로 되돌리는 작업 (화면에 잘 보이게)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    sample_img = std * sample_img + mean
    
    plt.imshow(sample_img)
    plt.title(f"Label: {labels[0].item()}")
    plt.show()