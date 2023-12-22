from importlib import import_module
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import argparse
import random
import os
from loss import create_criterion
from pathlib import Path
import re
import glob

# fix random seeds for reproducibility
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def main(args):
    # Seed 설정 
    seed_everything(args.seed)
    # Cuda 설정 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # GPU에서 연산
    # Save point 
    save_dir = increment_path(os.path.join(args.model_dir, args.name))
    
    # DataFrame 생성 
    data_loader_module = getattr(import_module("data_loader"), args.data_loader)
    data_loader = data_loader_module(base_dir = args.data_dir)
    df = data_loader.create_dataframe()
    
    # Train, Val data 분할 
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[args.target])
    
    # Transform 설정 
    custom_augmentation_module = getattr(import_module("dataset"), args.augmentation) # for training
    transform = custom_augmentation_module(args.resize)
    basic_augmentation_module = getattr(import_module("dataset"), "BasicAugmentation") # for val
    basic_transform = basic_augmentation_module(args.resize)
    
    custom2_augmentation_module = getattr(import_module("dataset"), "CustomAugmentation2") # for val
    custom2_transform = custom2_augmentation_module(args.resize)
    # face_augmentation_module = getattr(import_module("dataset"), "FaceAugmentation") # for val
    # face_transform = face_augmentation_module(args.resize)
    
    
    
    # Dataset 생성
    custom_dataset_module = getattr(import_module("dataset"), args.dataset) 
    train_dataset = custom_dataset_module(train_df,transform)
    train_dataset2 = custom_dataset_module(train_df,basic_transform)
    train_dataset3 = custom_dataset_module(train_df,custom2_transform)
    
    train_dataset = train_dataset + train_dataset2 + train_dataset3
    
    val_dataset = custom_dataset_module(val_df,basic_transform)
    # val_dataset = custom_dataset_module(val_df,basic_transform)
    
    # Data Loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available()) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    
    # Define Model
    model_module = getattr(import_module("model"), args.model) 
    model = model_module(num_classes=18)
    model = torch.nn.DataParallel(model)
    
    # Train set 
    criterion = create_criterion(args.criterion)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        # weight_decay=5e-5,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, threshold_mode='abs', min_lr=1e-8, verbose=True)
    trainer_module = getattr(import_module("trainer"), args.trainer)
    trainer = trainer_module(model= model,optimizer= optimizer, criterion = criterion, train_loader = train_loader, 
                             val_loader = val_loader, scheduler = scheduler, device = device, args = args, save_dir = save_dir)
    trainer.train()
    # print("success")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--data_loader", type=str, default="DataLoader", help="load data and make dataframe"
    )
    parser.add_argument(
        "--target", type=str, default="Total_label", help="target label, ex) Mask_label, Gender_label, Age_label ..."
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset", type=str, default="CustomDataset", help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation", type=str, default="CustomAugmentation", help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize", nargs=2, type=int, default=(224, 224), help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size", type=int, default=1000, help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="MyModel", help="model type (default: CustomModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="AdamW", help="optimizer type (default:AdamW)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion", type=str, default="cross_entropy", help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step", type=int, default=20, help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval", type=int, default=20, help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--trainer", type=str, default="Trainer", help="trainer for train",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )
    

    # Container environment
    parser.add_argument(
        "--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "../../Data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()
    # print(args)
    main(args)
    
    # /opt/conda/bin/python /data/ephemeral/home/level1-imageclassification-cv-05/EHmin/MyBase/main.py