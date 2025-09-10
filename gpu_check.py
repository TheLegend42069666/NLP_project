import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('device_name', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)