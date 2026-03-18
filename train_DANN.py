import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import cycle
import numpy as np

from model_DANN import get_model # 🌟 DANN 전용 모델 호출!
from data_loader import get_loader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 데이터 로더 (MMD와 동일하게 투트랙)
    train_loader = get_loader(f"{args.source}_train", batch_size=32)
    target_train_loader = get_loader(f"{args.target}_train", batch_size=32)
    val_loader = get_loader(f"{args.target}_val", batch_size=32)

    # 2. DANN 모델 & 옵티마이저
    model = get_model(num_classes=31).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 3. 채점관 (둘 다 CrossEntropy 사용)
    loss_class = nn.CrossEntropyLoss() # 정답 맞히기용
    loss_domain = nn.CrossEntropyLoss() # 도메인 속이기용

    for epoch in range(args.epochs):
        model.train()
        
        # 🌟 DANN의 핵심: 학습 진척도에 따라 alpha 조절 (0 -> 1)
        # 초반엔 정답 학습 위주, 후반엔 도메인 혼란 위주!
        p = float(epoch) / args.epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        pbar = tqdm(zip(train_loader, cycle(target_train_loader)), total=len(train_loader))
        for (s_img, s_label), (t_img, _) in pbar:
            s_img, s_label = s_img.to(device), s_label.to(device)
            t_img = t_img.to(device)
            
            # --- 도메인 라벨 생성 ---
            # Source는 0번, Target은 1번 경찰관에게 알려줄 정답지야!
            batch_size = s_img.size(0)
            domain_s_label = torch.zeros(batch_size).long().to(device)
            domain_t_label = torch.ones(t_img.size(0)).long().to(device)

            optimizer.zero_grad()

            # 🚀 STEP 1: Source 데이터 학습 (정답도 맞히고, 도메인도 판별받고)
            class_output, domain_output = model(s_img, alpha=alpha)
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(domain_output, domain_s_label)

            # 🚀 STEP 2: Target 데이터 학습 (도메인만 판별받음 - 정답은 모르니까!)
            _, domain_output = model(t_img, alpha=alpha)
            err_t_domain = loss_domain(domain_output, domain_t_label)

            # 🚀 STEP 3: 모든 에러 합산 후 역전파
            # 여기서 GRL 마법이 일어나서 특징 추출기는 도메인 판별기를 속이도록 깎여나감!
            total_loss = err_s_label + err_s_domain + err_t_domain
            total_loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1} [Loss: {total_loss.item():.4f}, Alpha: {alpha:.2f}]")

        # --- 검증 단계 (Validation) ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs) # alpha를 안 주면 정답만 뱉음
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    main(parser.parse_args())