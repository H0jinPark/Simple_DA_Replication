import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    # 클래스 폴더들 (backpack, bike, ...) 가져오기
    classes = os.listdir(source_dir)
    
    for cls in classes:
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        
        # 각 클래스별 이미지 파일 리스트
        src_cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(src_cls_path) if os.path.isfile(os.path.join(src_cls_path, f))]
        
        # 8:2 분할 (고정된 결과를 위해 random_state 사용)
        train_imgs, val_imgs = train_test_split(images, train_size=split_ratio, random_state=42)
        
        # 파일 복사
        for img in train_imgs:
            shutil.copy(os.path.join(src_cls_path, img), os.path.join(train_dir, cls, img))
        for img in val_imgs:
            shutil.copy(os.path.join(src_cls_path, img), os.path.join(val_dir, cls, img))

    print(f"✅ {os.path.basename(source_dir)} 분할 완료!")

# 실행 예시 (경로는 호진이의 실제 환경에 맞춰서 수정해줘!)
split_dataset('amazon', 'amazon_train', 'amazon_val')
split_dataset('webcam', 'webcam_train', 'webcam_val')
split_dataset('dslr', 'dslr_train', 'dslr_val')