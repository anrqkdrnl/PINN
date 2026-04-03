import torch

# True가 나오면 성공입니다!
print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")

# 내 그래픽 카드 이름 확인
print(f"사용 중인 장치: {torch.cuda.get_device_name(0)}")