import os
from pathlib import Path
from rembg import remove
from PIL import Image
import io

def remove_background(input_path, output_path):
    with open(input_path, 'rb') as input_file:
        input_bytes = input_file.read()
    output_bytes = remove(input_bytes)
    with open(output_path, 'wb') as output_file:
        output_file.write(output_bytes)

# 상위 디렉토리 경로
base_dir = Path('../Data')

# 입력 이미지와 출력 이미지 폴더 경로
input_dir = base_dir / 'train/images'
# input_dir = base_dir / 'testdir'
output_dir = base_dir / 'RemoveBgImages'

# output_dir이 없으면 생성
output_dir.mkdir(parents=True, exist_ok=True)

# input_dir 및 하위 디렉토리 내의 모든 jpg 파일에 대해 반복 처리
for input_path in input_dir.rglob('*.jpg') :
    # 각 파일에 대한 상대 경로 생성
    if input_path.name.startswith('._'):
        continue
    relative_path = input_path.relative_to(input_dir)
    output_path = output_dir / relative_path.with_suffix('.png')

    # output_path의 부모 디렉토리 생성
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 배경 제거 함수 실행
    remove_background(str(input_path), str(output_path))

print(f"배경이 제거된 이미지가 저장되었습니다: {output_path}")
