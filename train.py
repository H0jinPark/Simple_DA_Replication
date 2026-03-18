import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm

from model import get_model
from data_loader import get_loader

def main(args):
    # 1. 훈련장 위치 선정 (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 훈련이 진행될 장치: {device}") 
    
    print(f"🚀 학습 시작! Source: {args.source} | Target: {args.target}")
    print(f"📈 학습 에폭 수: {args.epochs}")

    # 2. 덤프트럭(DataLoader) 실전 배치!
    train_loader = get_loader(domain_name=f"{args.source}_train", batch_size=32)
    val_loader = get_loader(domain_name=f"{args.target}_val", batch_size=32)

    # 3. 모델 등판 및 훈련장 입장!
    model = get_model(num_classes=31)
    model = model.to(device)

    # 4. 채점관(Loss)과 학습 방향(Optimizer) 고용
    criterion = nn.CrossEntropyLoss()
    # 🌟 수정 포인트: args.lr 적용
    optimizer = optim.Adam(model.parameters(), lr=args.lr) 
    best_acc = 0.0

    # 🌟 수정 포인트: args.epochs 적용
    for epoch in range(args.epochs): 
        # --- [Train Mode] ---
        model.train() 
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")

        # --- [Eval Mode] ---
        model.eval() 
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_accuracy = 100 * correct / total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # 최고 점수 갱신 시 모델 저장
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("최고 점수 갱신! 모델 저장 완료\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Office-31 도메인 적응 실험")
    
    # 🌟 수정 포인트: source와 target으로 인자 통일!
    parser.add_argument('--source', type=str, required=True, help='Source 도메인 (예: amazon)')
    parser.add_argument('--target', type=str, required=True, help='Target 도메인 (예: amazon, webcam)')
    parser.add_argument('--epochs', type=int, default=5, help='학습 에폭 수 (기본값: 5)')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률 (기본값: 0.0001)')

    args = parser.parse_args()
    main(args)