import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

from dataset import * #MaskBaseDataset
from loss import create_criterion

from sklearn.model_selection import train_test_split
import pandas as pd


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


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


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), "MaskBaseDataset")  # default: MaskBaseDataset
    dataset = dataset_module(data_dir=data_dir, )
    num_classes = dataset.num_classes  # 18
    
    # changed phase
    ################################################################
    
    # 데이터 불러와서 label 붙이기
    df = pd.DataFrame({'img_path' : dataset.image_paths, 'label' :dataset.multi_class_labels})
    df['age'] = df['img_path'].apply(lambda x: x.split('/')[-2][-2:])

    ignore_age = df[('55'< df['age']) & (df['age'] <'60')]
    df = df[(55 >= df['age'].astype(int)) | (df['age'].astype(int) >= 60)]

    train_df, val_df, _, _ = train_test_split(df, df['label'].values, test_size=args.val_ratio, random_state=args.seed, stratify=df['label'].values)
    train_df_young_age_all = train_df[train_df['label'].isin([0, 3, 6, 9, 12, 15])]   
    train_df_middle_age_male = train_df[train_df['label'].isin([1, 7, 13])] 
    train_df_middle_age_female = train_df[train_df['label'].isin([4, 10, 16])]  
    train_df_old_age_all = train_df[train_df['label'].isin([2, 5, 8, 11, 14, 17])]  

    # augmentation에 전달
    transform_module = getattr(import_module("dataset"), "None_aug") 
    None_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    transform_module = getattr(import_module("dataset"), "Horizontal_Rotate_aug") 
    Horizontal_Rotate_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    transform_module = getattr(import_module("dataset"), "Rotate_aug") 
    Rotate_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    transform_module = getattr(import_module("dataset"), "ColorJitter_Flip_aug") 
    ColorJitter_Flip_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    transform_module = getattr(import_module("dataset"), "ColorJitter_aug") 
    ColorJitter_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    transform_module = getattr(import_module("dataset"), "Grayscale_aug") 
    Grayscale_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    transform_module = getattr(import_module("dataset"), "Sharpness_aug") 
    Sharpness_aug = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std        
    )

    
    # 그룹별 이미지 주소와 label 받아오기
    train_young_age_all_path, train_young_age_all_label        = train_df_young_age_all["img_path"].values,      train_df_young_age_all["label"].values
    train_middle_age_male_path, train_middle_age_male_label   = train_df_middle_age_male["img_path"].values,     train_df_middle_age_male["label"].values
    train_middle_age_female_path, train_middle_age_female_label = train_df_middle_age_female["img_path"].values, train_df_middle_age_female["label"].values
    train_old_age_all_path, train_old_age_all_label            = train_df_old_age_all["img_path"].values,        train_df_old_age_all["label"].values

    train_dataset = []
    
    # 원본 이미지
    train_dataset.append(CustomDataset(train_young_age_all_path, train_young_age_all_label, None_aug))
    train_dataset.append(CustomDataset(train_middle_age_male_path, train_middle_age_male_label, None_aug))
    train_dataset.append(CustomDataset(train_middle_age_female_path, train_middle_age_female_label, None_aug))
    train_dataset.append(CustomDataset(train_old_age_all_path, train_old_age_all_label, None_aug))

    #중년 남성 증강 3배
    train_dataset.append(CustomDataset(train_middle_age_male_path, train_middle_age_male_label, Horizontal_Rotate_aug))
    train_dataset.append(CustomDataset(train_middle_age_male_path, train_middle_age_male_label, ColorJitter_Flip_aug))

    #중년 여성 증강 2배
    train_dataset.append(CustomDataset(train_middle_age_female_path, train_middle_age_female_label, Horizontal_Rotate_aug))

    #노년 남성/여성 증강 6배
    train_dataset.append(CustomDataset(train_old_age_all_path, train_old_age_all_label, Horizontal_Rotate_aug))
    train_dataset.append(CustomDataset(train_old_age_all_path, train_old_age_all_label, ColorJitter_Flip_aug))
    train_dataset.append(CustomDataset(train_old_age_all_path, train_old_age_all_label, ColorJitter_aug))
    train_dataset.append(CustomDataset(train_old_age_all_path, train_old_age_all_label, Grayscale_aug))
    train_dataset.append(CustomDataset(train_old_age_all_path, train_old_age_all_label, Sharpness_aug))

    train_set = ConcatDataset(train_dataset)

    # valid data
    val_img_paths, val_labels = val_df['img_path'].values, val_df['label'].values
    val_set = CustomDataset(val_img_paths, val_labels, None_aug)
    
    #인덱스 잘 받아오는지 확인    
    print("_____df_______ :\n", df)
    print("__ignore_age___ :\n",ignore_age)
    print("__55세 출력__",df[df['age'].astype(int)==55])
    print("__56세 출력__",df[df['age'].astype(int)==56])
    print("__57세 출력__",df[df['age'].astype(int)==57])
    print("__58세 출력__",df[df['age'].astype(int)==58])
    print("__59세 출력__",df[df['age'].astype(int)==59])
    print("___청년________\n", len(train_df_young_age_all)//7, train_df_young_age_all.head())
    print("___중년 남성___\n", len(train_df_middle_age_male)//7, train_df_middle_age_male.head())
    print("___중년 여성___\n", len(train_df_middle_age_female)//7,train_df_middle_age_female.head())
    print("___노년________\n", len(train_df_old_age_all)//7, train_df_old_age_all.head())
    print("__total__: ", len(train_set)+len(val_set) , ",  __train__ : ", len(train_set), ",  __val__ : ",len(val_set))
    
    #train : 15120,  val : 3780, total : 18900, changed : 29658

    ################################################################

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
            
    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.2)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    
    for epoch in range(args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch+1}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = (torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy())
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=True
                        )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--epochs", type=int, default=6, help="number of epochs to train (default: 1)")
    #parser.add_argument("--dataset", type=str, default="MaskBaseDataset", help="dataset augmentation type (default: MaskBaseDataset)",)
    #parser.add_argument("--augmentation", type=str, default="BaseAugmentation", help="data augmentation type (default: BaseAugmentation)",)
    parser.add_argument("--resize", nargs=2, type=int, default=[224, 224], help="resize size for image when training",)
    parser.add_argument("--batch_size", type=int, default=64, help="input batch size for training (default: 64)",)
    parser.add_argument("--valid_batch_size", type=int, default=1000, help="input batch size for validing (default: 1000)",)
    parser.add_argument("--model", type=str, default="swin_tranformer", help="model type (default: BaseModel)")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer type (default: SGD)")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate (default: 1e-3)")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="ratio for validaton (default: 0.2)",)
    parser.add_argument("--criterion", type=str, default="focal", help="criterion type (default: cross_entropy)",)
    parser.add_argument("--lr_decay_step", type=int, default=2, help="learning rate scheduler deacy step (default: 20)",)
    parser.add_argument("--log_interval", type=int, default=20, help="how many batches to wait before logging training status",)
    parser.add_argument("--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}")

    # Container environment
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/data/ephemeral/home/level1-imageclassification-cv-05/Data/train/images"),)
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
