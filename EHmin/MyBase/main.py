from importlib import import_module
import torch
import numpy as np

print(torch.version.cuda)

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main():
    # Cuda 설정 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataFrame 생성 
    data_loader_module = getattr(import_module("data_loader"), "DataLoader")
    data_loader = data_loader_module(base_dir = './Data/train/images/') #/ Configparser로 만들자!
    df = data_loader.create_dataframe()
    
    # # Transform 설정 
    transform = getattr(import_module("dataset"), "transform")
    
    # Dataset 생성
    custom_dataset_module = getattr(import_module("dataset"), "CustomDataset")
    custom_dataset = custom_dataset_module(df,transform)
    
    print("success")
    

if __name__ == '__main__':
    main()