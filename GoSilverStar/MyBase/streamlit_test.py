import streamlit as st
import os
import torch
from PIL import Image
from torchvision import transforms
from importlib import import_module
import json

# 기존의 load_config, load_model 등의 함수들

def get_label(prediction):
    # 클래스 인덱스를 실제 레이블로 매핑
    labels = [
        "Wear Male < 30", "Wear Male >= 30 and < 60", "Wear Male >= 60",
        "Wear Female < 30", "Wear Female >= 30 and < 60", "Wear Female >= 60",
        "Incorrect Male < 30", "Incorrect Male >= 30 and < 60", "Incorrect Male >= 60",
        "Incorrect Female < 30", "Incorrect Female >= 30 and < 60", "Incorrect Female >= 60",
        "Not Wear Male < 30", "Not Wear Male >= 30 and < 60", "Not Wear Male >= 60",
        "Not Wear Female < 30", "Not Wear Female >= 30 and < 60", "Not Wear Female >= 60"
    ]
    return labels[prediction]

def load_model_for_streamlit(saved_model, num_classes, device):
    # 모델 클래스를 불러옵니다. 'BaseModel'은 예시 이름입니다.
    model_cls = getattr(import_module("model"), "MyModel")
    model = model_cls(num_classes=num_classes)

    # 모델의 가중치를 불러옵니다.
    model_path = os.path.join(saved_model, "best.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


def predict(image, model, device):
    # 이미지를 모델에 맞는 형식으로 변환합니다. 
    # 이 예제에서는 이미지를 리사이즈하고 텐서로 변환하는 과정을 포함합니다.
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 모델에 맞는 크기로 조정
        transforms.ToTensor(),  # PIL 이미지를 텐서로 변환
    ])
    image = transform(image).unsqueeze(0).to(device)  # 배치 차원 추가 및 장치 할당

    # 모델을 사용하여 예측
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    # return predicted.item()
    predicted_label = get_label(predicted.item())
    
    return predicted_label



# Streamlit 웹 인터페이스
def main():
    st.title("이미지 분류")

    # 파일 업로더
    uploaded_file = st.file_uploader("이미지를 선택하세요...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')

        # 업로드된 이미지 표시
        st.image(image, caption='업로드된 이미지.', use_column_width=True)
        st.write("")
        st.write("분류 중...")

        # 모델 준비
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dir = "./model/exp30"  # 경로는 필요에 따라 조정하세요
        num_classes = 18  # 필요에 따라 조정하세요
        model = load_model_for_streamlit(model_dir, num_classes, device)
        model.to(device)
        model.eval()

        # 예측
        prediction = predict(image, model, device)

        # 예측 결과 표시
        st.write(f"예측: {prediction}")

if __name__ == "__main__":
    main()
