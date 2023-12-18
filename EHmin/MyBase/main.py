from importlib import import_module
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# fix random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
# os.environ['PYTHONHASHSEED'] = 'a'


def main():
    # Cuda 설정 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # DataFrame 생성 
    data_loader_module = getattr(import_module("data_loader"), "DataLoader")
    data_loader = data_loader_module(base_dir = './Data/train/images/') #/ Configparser로 만들자!
    df = data_loader.create_dataframe()
    
    # Train, Val data 분할 
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Total_label']) #/ Configparser로 만들자!
    
    # Transform 설정 
    custom_augmentation_module = getattr(import_module("dataset"), "CustomAugmentation") #/ Configparser로 만들자!
    transform = custom_augmentation_module((256,256))
    basic_augmentation_module = getattr(import_module("dataset"), "BasicAugmentation")
    basic_transform = basic_augmentation_module((256,256))
    
    # Dataset 생성
    custom_dataset_module = getattr(import_module("dataset"), "CustomDataset") #/ Configparser로 만들자!
    train_dataset = custom_dataset_module(train_df,transform)
    val_dataset = custom_dataset_module(val_df,basic_transform)
    
    # Data Loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0) #/ Configparser로 만들자!
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    model_module = getattr(import_module("model"), "CustomModel") #/ Configparser로 만들자!
    model = model_module(num_classes=18).to(device)  #/ Configparser로 만들자!
    model = torch.nn.DataParallel(model)
    print(model)
    
    # print("success")


if __name__ == '__main__':
    main()