import argparse
import multiprocessing
import os
from importlib import import_module
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
import pickle

# from dataset import TestDataset, MaskBaseDataset

def load_config(model_dir):
    """
    model_dir 경로에서 config.json 파일을 읽고 해당 내용을 반환합니다.

    Args:
        model_dir (str): 모델 디렉토리 경로

    Returns:
        dict: config.json 파일의 내용
    """
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as file:
        return json.load(file)


def load_model(saved_model, num_classes, device):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, args):
    """
    모델 추론을 수행하는 함수

    Args:
        data_dir (str): 테스트 데이터가 있는 디렉토리 경로
        model_dir (str): 모델 가중치가 저장된 디렉토리 경로
        output_dir (str): 결과 CSV를 저장할 디렉토리 경로
        args (argparse.Namespace): 커맨드 라인 인자

    Returns:
        None
    """
    
    # CUDA를 사용할 수 있는지 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir,18, device).to(device)
    model.eval()

    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    df = pd.DataFrame(img_paths, columns=['Image_path'])
    basic_augmentation_module = getattr(import_module("dataset"), "BasicAugmentation") # for val
    basic_transform = basic_augmentation_module(args.resize)
    
    dataset = CustomDataset(df, basic_transform, test= True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            # 정답을 만들때는 반드시 아래의 주석을 풀어줄 것!!!. 앙상블을 위해서 아래의 코드를 주석처리함!
            # if not (args.ensemble_boolean) : 
            # pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info["ans"] = preds
    save_path = os.path.join(model_dir, f"output.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")

    # 리스트를 pickle 파일로 저장
    save_path = os.path.join(model_dir, f"my_list.pkl")
    with open(save_path, 'wb') as file:
        pickle.dump(preds, file)

    print("List saved to pickle file.")

if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./model/exp"),
    )
    
    pre_args = parser.parse_args()
    
    config = load_config(pre_args.model_dir)
    parser.set_defaults(**config) # 기존 config 파일의 정보로 학습 시작 

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--model", type=str, help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--ensemble_boolean",
        action='store_true',
        help="Set this flag to enable ensemble (default: False)"
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "/data/ephemeral/home/level1-imageclassification-cv-05/Data/eval"),
    )
    # parser.add_argument(
    #     "--output_dir",
    #     type=str,
    #     default=os.environ.get("SM_OUTPUT_DATA_DIR", "./output"),
    # )

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    # output_dir = args.output_dir

    # os.makedirs(output_dir, exist_ok=True)

    # 모델 추론을 수행한다.
    inference(data_dir, model_dir, args)
    
    #/opt/conda/bin/python /data/ephemeral/home/level1-imageclassification-cv-05/EHmin/MyBase/inference.py --model_dir=./model/exp
