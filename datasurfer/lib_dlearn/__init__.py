import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    n_cudas = torch.cuda.device_count()
    for i in range(n_cudas):
        print(torch.cuda.get_device_name(i))