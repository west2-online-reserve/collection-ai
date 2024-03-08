# from torchvision.datasets.utils import extract_archive
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import pandas as pd
# 划分数据集
data_dir = "/kaggle/input/mycaltech101/101_ObjectCategories"
# annotations_file = "/kaggle/output/working/annotations.csv"
annotations_file = "./annotations.csv"

with open(annotations_file,'w') as f:
    for root,dirs,files in os.walk(data_dir):
        for file in files:
            label = os.path.basename(root)
            img_path = os.path.join(root,file)
            f.write(f"{img_path},{label}\n")
# 划分训练集和测试集
df = pd.read_csv(annotations_file, header=None)
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[1], random_state=42)

train_df.to_csv("./train_annotations.csv", index=False, header=False)
test_df.to_csv("./test_annotations.csv", index=False, header=False)