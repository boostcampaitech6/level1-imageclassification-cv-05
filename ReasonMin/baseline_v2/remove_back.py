from rembg.bg import remove
import io
from PIL import Image
import os

# back_removed_img 폴더가 없으면 새로 생성
img_path = '/data/ephemeral/home/level1-imageclassification-cv-05/Data/train/images'
removed_img_path = '/data/ephemeral/home/level1-imageclassification-cv-05/Data/train/back_removed_img'
os.makedirs(removed_img_path, exist_ok=True)

# img_path의 하위 폴더 읽기
profiles = sorted(os.listdir(img_path))
compare_file = os.listdir(removed_img_path)
_profiles = [folder for folder in profiles if not folder.startswith("._")]

print("__기존 데이터 개수__: ", len(profiles))
print("__실제 데이터 개수__: ", len(_profiles))
print("__변형 데이터 개수__: ", len(compare_file))
num = len(compare_file)


for profile in _profiles:
    if profile.startswith("."):  # "." 로 시작하는 폴더은 무시합니다
        continue
    
    # profile 폴더가 없거나, 이미 만들어진 폴더에 img가 7개 되지 않는다면 실행
    try:
        detect_folder = os.path.join(removed_img_path,profile)
        files_num = len(os.listdir(detect_folder))
        if files_num == 7:
            continue
    except FileNotFoundError:

        img_folder = os.path.join(img_path, profile)

        # 폴더 removed_img_path 아래에 같은 폴더 만들기
        os.makedirs(os.path.join(removed_img_path, profile), exist_ok=True)

        # 하위 폴더 열기
        for image in os.listdir(img_folder):
            if image.startswith("."):   # "." 로 시작하는 파일은 무시합니다
                continue

            each_img_path = os.path.join(img_path, profile, image)
            ch_4_img = remove(Image.open(each_img_path))           #remove를 통과하면 4채널이다.
            get_img = ch_4_img.convert("RGB")                      # 3채널로 돌려준다
            
            # 새로운 폴더에 동일한 폴더명으로 저장
            output_img_path = os.path.join(removed_img_path, profile, image)
            get_img.save(output_img_path)
        
        num += 1
        print("데이터 수 :",num, "--create new folder--", profile)
        

print(" -- 완료 -- ")