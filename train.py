import torch
import torch.nn as nn
import torch.optim as optim

# 🌟 1. 우리가 직접 깎은 보물들 불러오기! 🌟
from model import get_model
from data_loader import get_loader  # <-- 드디어 덤프트럭 등장!
from tqdm import tqdm

# 2. 훈련장 위치 선정 (GPU vs CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 훈련이 진행될 장치: {device}")

# 3. 덤프트럭(DataLoader) 실전 배치!
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"현재 훈련이 진행될 장치: {device}") 
    # (경로는 호진 님 폴더 구조에 맞춰서 'amazon', 'webcam' 등으로 지정해주세요)
    train_loader = get_loader(domain_name='amazon', batch_size=32)
    val_loader = get_loader(domain_name='webcam', batch_size=32)
    print("🚛 훈련 데이터, 검증 데이터 로드 완료!")

    # 4. 모델 등판 및 훈련장 입장!
    model = get_model(num_classes=31)
    model = model.to(device)

    # 3. 채점관(Loss)과 학습 방향(Optimizer) 고용
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_acc = 0.0

    # 에포크(Epoch): 전체 데이터 트럭(1회독)을 몇 번 반복해서 볼 것인가?
    num_epochs = 1

    for epoch in range(num_epochs):
        # 0. 모델에게 "지금은 실전(Test)이 아니라 훈련(Train)이야!" 라고 선전포고
        model.train() 
        running_loss = 0.0
        
        # 덤프트럭(dataloader)에서 32장씩(batch) 묶어서 꺼내옵니다!
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in pbar:
            
            # 1. 데이터도 훈련장(GPU)으로 올려보내기!
            inputs, labels = inputs.to(device), labels.to(device)
            # --- 🌟 파이토치 학습의 절대 5원칙 🌟 ---
            
            # 2. 내비게이션(기울기) 초기화! (이전 문제의 잔재 지우기)
            optimizer.zero_grad()
            
            # 3. 모델에게 문제 풀게 하기 (Forward)
            outputs = model(inputs)
            
            # 4. 채점관이 정답과 비교해서 점수 매기기 (Loss)
            loss = criterion(outputs, labels)
            
            # 5. 오답 노트 작성 및 원인 분석 (Backward)
            loss.backward()
            
            # 6. 모델 뇌세포(가중치) 업데이트 (Step)
            optimizer.step()
            
            # --------------------------------------
            
            # 이번 배치의 오차(Loss)를 기록해둡니다.
            running_loss += loss.item()
            
        # 1 에포크(전체 데이터 1회독)가 끝날 때마다 평균 오차를 출력!
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # 0. 모델에게 "이제 실전 모의고사야! 긴장해!" 라고 모드 전환
        model.eval() 
        
        correct = 0
        total = 0
        
        # 1. 모의고사 볼 때는 오답 노트(기울기 계산)를 쓸 필요가 없으니, 
        # 파이토치에게 "계산기 꺼둬!"라고 명령 (메모리, 속도 대폭 향상)
        with torch.no_grad():
            
            # 이번엔 검증용 덤프트럭(val_loader)에서 처음 보는 데이터를 꺼내옵니다.
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 2. 문제 풀기 (Forward)
                outputs = model(inputs)
                
                # 3. 31개 보기 중에서 모델이 가장 확신하는(확률이 높은) 정답 번호 고르기
                _, predicted = torch.max(outputs, 1)
                
                # 4. 채점하기 (총 몇 문제 풀었고, 몇 문제 맞혔는지 누적)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # 5. 이번 에포크의 모의고사 정답률(Accuracy) 계산
        val_accuracy = 100 * correct / total
        print(f"👉 모의고사 정답률: {val_accuracy:.2f}%")

        # 6. (보너스) 역대급 점수가 나왔다면? 모델의 뇌(가중치)를 파일로 박제!
        # (코드 맨 위에 best_acc = 0.0을 미리 만들어뒀다고 가정합니다)
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            # 모델의 뇌 구조(state_dict)를 .pth 파일로 저장합니다.
            torch.save(model.state_dict(), "best_model.pth")
            print("🎉 최고 점수 갱신! 모델을 저장했습니다. 🎉\n")