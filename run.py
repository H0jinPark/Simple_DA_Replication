import os

# 돌리고 싶은 5가지 실험 명령어들을 리스트로 쭉 적어줍니다.
commands = [
    # # 1. Base 모델 Upper Bound (천장) 확인
    # "python train.py --source amazon --target amazon --epochs 10",
    
    # # 2. Base 모델 Lower Bound (바닥) 확인
    # "python train.py --source amazon --target webcam --epochs 10",
    # "python train.py --source amazon --target dslr --epochs 10",
    
    # # 3. MMD 모델 성능 복구 확인
    # "python train_MMD.py --source amazon --target webcam --epochs 10",
    # "python train_MMD.py --source amazon --target dslr --epochs 10"

    # "python train_DANN.py --source amazon --target webcam --epochs 10",
    "python train_DANN.py --source amazon --target dslr --epochs 10"
]

for idx, cmd in enumerate(commands):
    print(f"\n{'='*60}")
    print(f"🚀 [실험 {idx+1}/{len(commands)}] 시작: {cmd}")
    print(f"{'='*60}\n")
    
    # 터미널에 명령어를 입력하고 실행하는 핵심 코드!
    # 이 실험이 완전히 끝날 때까지 기다렸다가 다음 루프로 넘어갑니다.
    os.system(cmd)
    
print("\n실험이 모두 끝났습니다!")