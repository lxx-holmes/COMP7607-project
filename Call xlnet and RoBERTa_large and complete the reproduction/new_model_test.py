from transformers import BertForSequenceClassification, BertTokenizer, RobertaTokenizer, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, XLNetForSequenceClassification, XLNetTokenizer
import pandas as pd
import os
import sys
from time import time, sleep
from transformers import BertForSequenceClassification, BertTokenizerFast, RobertaTokenizerFast, RobertaForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, XLNetForSequenceClassification, XLNetTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score
import numpy as np

from transformers import XLNetTokenizerFast, XLNetForSequenceClassification
import pandas as pd
import os


# 模型路径
model_path = "C:/Users/11570/Desktop/7607 final project/fomc-hawkish-dovish/code_model/Weiqi"

# 数据路径
data_path = "C:/Users/11570/Desktop/7607 final project/fomc-hawkish-dovish/data/filtered_data"

# 输出路径
output_path = "C:/Users/11570/Desktop/7607 final project/fomc-hawkish-dovish/data/filtered_data"

# 文件夹名称列表
folder_names = ["meeting_minutes", "press_conference", "speech"]

# 加载模型和分词器
tokenizer = XLNetTokenizerFast.from_pretrained(model_path, do_lower_case=True, do_basic_tokenize=True)
model = XLNetForSequenceClassification.from_pretrained(model_path, num_labels=3)

# Set max length
max_length = 256

# Classification function
'''def classify_sentences(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    logits = model(**inputs).logits
    predicted_classes = logits.argmax(dim=1).tolist()
    return predicted_classes'''
def classify_sentences(sentences):
    if not sentences:
        return []  # Return an empty list if there are no sentences
    
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    if 'input_ids' not in inputs or 'attention_mask' not in inputs:
        return []  # Return an empty list if tokenization fails or inputs are invalid
    
    logits = model(**inputs).logits
    predicted_classes = logits.argmax(dim=1).tolist()
    return predicted_classes

# Iterate over each folder
for folder_name in folder_names:
    folder_path = os.path.join(data_path, folder_name)
    output_folder_path = os.path.join(output_path, f"{folder_name}_labeled")
    
    # Create a new folder to store processed files
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Iterate over CSV files in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                # Read CSV file
                df = pd.read_csv(file_path)
                sentences = df["sentence"].tolist()
                
                # Classify sentences and get labels
                labels = classify_sentences(sentences)
                
                # Add labels to the DataFrame
                df["label"] = labels
                
                # Save the labeled DataFrame to a new CSV file
                output_file = os.path.join(output_folder_path, f"labeled_{file}")
                df.to_csv(output_file, index=False)