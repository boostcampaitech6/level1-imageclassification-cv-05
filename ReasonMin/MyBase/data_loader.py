import os
import pandas as pd

class DataLoader:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def create_dataframe(self):
        """_summary_ : DF를 만들어서 return 

        Returns:
            _type_: pandas DF
        """
        data = []
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                labels = item.split('_')
                for file in os.listdir(item_path):
                    if file.startswith('.'):
                        continue
                    if 'incorrect_mask' in file:
                        mask_label = 1
                    elif any(mask in file for mask in ['mask1', 'mask2', 'mask3', 'mask4', 'mask5']):
                        mask_label = 0
                    elif 'normal' in file:
                        mask_label = 2
                    else:
                        continue
                    gender_label = 0 if labels[1].lower() == "male" else 1
                    age_label = 0 if int(labels[3]) < 30 else (2 if int(labels[3]) >= 60 else 1)
                    total_label = 6 * mask_label + 3 * gender_label + age_label
                    data.append({
                        'Image_path': os.path.join(item_path, file),
                        'Mask_label': mask_label,
                        'Gender_label': gender_label,
                        'Age_label': age_label,
                        'Total_label': total_label,
                        'age':labels[3]
                    })
        df = pd.DataFrame(data)
        return df

    def save_dataframe_to_csv(self, df, file_path='./dataframe.csv'):
        df.to_csv(file_path, index=False)