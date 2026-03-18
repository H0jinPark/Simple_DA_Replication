import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from itertools import cycle

from model_MMD import get_model
from data_loader import get_loader

class MMDLoss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        
        # 총 거리 행렬 계산
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.gaussian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        
        # XX, YY, XY, YX 로 나누어서 MMD 거리 계산
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return loss
    
def main(args):
    # 1. 훈련장 위치 선정 (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 훈련이 진행될 장치: {device}") 
    
    print(f"🚀 MMD 학습 시작! Source: {args.source} | Target: {args.target}")
    print(f"📈 학습 에폭 수: {args.epochs}")

    # 2. 덤프트럭(DataLoader) 실전 배치! (투트랙 🌟)
    # Source는 학습용(정답 O), Target은 적응용(정답 X), Target Val은 최종 시험용!
    train_loader = get_loader(domain_name=f"{args.source}_train", batch_size=32)
    target_train_loader = get_loader(domain_name=f"{args.target}_train", batch_size=32) # 🌟 추가됨
    val_loader = get_loader(domain_name=f"{args.target}_val", batch_size=32)

    # 3. 모델 등판 및 훈련장 입장!
    model = get_model(num_classes=31)
    model = model.to(device)

    # 4. 채점관(Loss)과 학습 방향(Optimizer) 고용
    criterion = nn.CrossEntropyLoss()
    mmd_loss_fn = MMDLoss() # 🌟 MMD 채점관 추가
    lambda_mmd = 0.5        # 🌟 MMD 가중치 (필요하면 인자로 빼도 좋아!)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    best_acc = 0.0

    for epoch in range(args.epochs): 
        # --- [Train Mode] ---
        model.train() 
        running_cls_loss = 0.0
        running_mmd_loss = 0.0
        
        # 🌟 zip과 cycle로 Source 트럭과 Target 트럭을 동시에 병렬로 부르기!
        pbar = tqdm(zip(train_loader, cycle(target_train_loader)), total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for (source_inputs, labels), (target_inputs, _) in pbar:
            # 둘 다 GPU로 올리기
            source_inputs, labels = source_inputs.to(device), labels.to(device)
            target_inputs = target_inputs.to(device)
            
            optimizer.zero_grad()
            
            # 🌟 모델이 2개의 값을 뱉어냅니다: (특징 벡터, 최종 예측값)
            source_features, source_outputs = model(source_inputs)
            target_features, _ = model(target_inputs) # 타겟은 정답을 맞출 필요가 없으니 예측값은 버림(_)
            
            # 🌟 두 가지 Loss 계산
            cls_loss = criterion(source_outputs, labels)              # 기본 분류 Loss
            mmd_loss = mmd_loss_fn(source_features, target_features)  # MMD 거리 Loss
            
            # 최종 Loss = 분류 Loss + (가중치 * MMD Loss)
            loss = cls_loss + (lambda_mmd * mmd_loss)
            
            loss.backward()
            optimizer.step()
            
            # 로그 출력을 위해 각각 누적
            running_cls_loss += cls_loss.item()
            running_mmd_loss += mmd_loss.item()
            
        epoch_cls_loss = running_cls_loss / len(train_loader)
        epoch_mmd_loss = running_mmd_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}] | Cls Loss: {epoch_cls_loss:.4f} | MMD Loss: {epoch_mmd_loss:.4f}")

        # --- [Eval Mode] ---
        model.eval() 
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 🌟 평가할 때도 모델이 2개를 뱉으므로 앞에 특징(feature)은 언더바(_)로 받아서 무시
                _, outputs = model(inputs) 
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # 최고 점수 갱신 시 모델 저장
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), "best_model_mmd.pth")
            print("🎉 최고 점수 갱신! MMD 모델 저장 완료\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Office-31 도메인 적응 실험")
    
    # 🌟 수정 포인트: source와 target으로 인자 통일!
    parser.add_argument('--source', type=str, required=True, help='Source 도메인 (예: amazon)')
    parser.add_argument('--target', type=str, required=True, help='Target 도메인 (예: amazon, webcam)')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에폭 수 (기본값: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률 (기본값: 0.0001)')

    args = parser.parse_args()
    main(args)