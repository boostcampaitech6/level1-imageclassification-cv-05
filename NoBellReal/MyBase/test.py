import os
import pandas as pd
import numpy as np
import pickle
from scipy.stats import mode
import argparse

# 기본 모델 디렉토리 경로
BASE_MODEL_DIRECTORIES = ['./model/exp', './model/exp2', './model/exp3']
PKL_FILE_PATHS = ['./model/exp/probs.pkl', './model/exp2/probs.pkl', './model/exp3/probs.pkl']

def load_pickle(file_path):
    """Pickle 파일에서 데이터를 로드합니다."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def soft_voting_from_pkl(*prob_files):
    """Pickle 파일로부터 로드된 확률 데이터를 사용하여 soft voting을 수행합니다."""
    probabilities = [load_pickle(file) for file in prob_files]
    avg_prob = np.mean(np.array(probabilities), axis=0)
    final_predictions = np.argmax(avg_prob, axis=1)
    return final_predictions

def main(args):
    if args.ensemble_method == 'soft_voting':
        # soft_voting: .pkl 파일에서 확률 데이터 로드
        final_predictions = soft_voting_from_pkl(*PKL_FILE_PATHS)

        # output_soft.csv 파일에서 이미지 이름 읽기

    else:
        # hard_voting: 기존 방식 유지
        file_name = 'output.csv'
        model_output_files = [os.path.join(directory, file_name) for directory in BASE_MODEL_DIRECTORIES]
        model_outputs = [pd.read_csv(file_path) for file_path in model_output_files]
        final_predictions = hard_voting(*model_outputs)
        image_ids = model_outputs[0]['ImageID']

    # 결과 파일 저장
    output_df = pd.DataFrame({'ImageID': image_ids, 'pred': final_predictions})
    output_file_path = os.path.join(os.getcwd(), 'ensemble_results.csv')
    output_df.to_csv(output_file_path, index=False)
    print(f"Ensemble results saved to {output_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_method', type=str, choices=['hard_voting', 'soft_voting'], help='Ensemble method to use', required=True)

    args = parser.parse_args()
    main(args)
