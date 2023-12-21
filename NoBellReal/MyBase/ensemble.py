import pandas as pd
import numpy as np
import os
import pickle
from scipy.stats import mode
import argparse

# 기본 모델 디렉토리 경로
BASE_MODEL_DIRECTORIES = ['./model/exp', './model/exp2', './model/exp3']

def load_pickle(file_path):
    """Pickle 파일에서 데이터를 로드합니다."""
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def read_model_output(file_path):
    return pd.read_csv(file_path)

def hard_voting(*outputs):
    votes = np.array([output['ans'].values for output in outputs])
    final_votes = mode(votes, axis=0)[0].flatten()
    return final_votes

def soft_voting(*prob_files):
    """Pickle 파일로부터 로드된 확률 데이터를 사용하여 soft voting을 수행합니다."""
    probabilities = [load_pickle(file) for file in prob_files]
    avg_prob = np.mean(np.array(probabilities), axis=0)
    final_predictions = np.argmax(avg_prob, axis=1)
    return final_predictions

def main(args):
    if args.ensemble_method == 'hard_voting':
        file_name = 'output.csv'
    else:
        file_name = 'my_list.pkl'

    model_output_files = [os.path.join(directory, file_name) for directory in BASE_MODEL_DIRECTORIES]

    if args.ensemble_method == 'hard_voting':
        model_outputs = [read_model_output(file_path) for file_path in model_output_files]
        final_predictions = hard_voting(*model_outputs)
    elif args.ensemble_method == 'soft_voting':
        final_predictions = soft_voting(*model_output_files)  # '*'를 사용하여 리스트를 개별 요소로 전달

    image_ids = pd.read_csv(os.path.join(BASE_MODEL_DIRECTORIES[0], 'output.csv'))['ImageID']
    output_df = pd.DataFrame({'ImageID': image_ids, 'ans': final_predictions})
    output_file_path = os.path.join(os.getcwd(), 'ensemble_results.csv')
    output_df.to_csv(output_file_path, index=False)
    print(f"Ensemble results saved to {output_file_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble_method', type=str, choices=['hard_voting', 'soft_voting'], help='Ensemble method to use', required=True)

    args = parser.parse_args()
    main(args)
