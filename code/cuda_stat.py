import torch

print(torch.cuda.device_count())  # 실제 접근 가능한 GPU 수
print(torch.cuda.current_device())  # 현재 기본 디바이스 인덱스
