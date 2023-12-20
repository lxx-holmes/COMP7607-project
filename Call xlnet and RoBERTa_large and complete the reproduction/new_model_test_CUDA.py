from transformers import XLNetForSequenceClassification, XLNetTokenizerFast
import pandas as pd
import os
import torch

# 模型路径
model_path = "C:/Users/11570/Desktop/7607 final project/fomc-hawkish-dovish/code_model/Weiqi"

# 数据路径
data_path = "C:/Users/11570/Desktop/7607 final project/fomc-hawkish-dovish/data/filtered_data"

# 输出路径
output_path = "C:/Users/11570/Desktop/7607 final project/fomc-hawkish-dovish/data/filtered_data"

# 文件夹名称列表
folder_names = ["meeting_minutes", "press_conference", "speech"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# 加载模型和分词器
tokenizer = XLNetTokenizerFast.from_pretrained(model_path, do_lower_case=True, do_basic_tokenize=True)
model = XLNetForSequenceClassification.from_pretrained(model_path, num_labels=3)
model.to(device)

# Set max length
max_length = 256

# 分类函数
def classify_sentences(sentences):
    if not sentences:
        return []  # 如果没有句子，则返回一个空列表
    
    # 将句子转换为张量，并将输入张量移动到CUDA设备上
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = {key: inputs[key].to(device) for key in inputs.keys()}
    
    if 'input_ids' not in inputs or 'attention_mask' not in inputs:
        return []  # 如果标记化失败或输入无效，则返回一个空列表
    
    # print("Model device:", next(model.parameters()).device)
    # print("Input tensor device:", inputs["input_ids"].device)
    
    logits = model(**inputs).logits
    predicted_classes = logits.argmax(dim=1).tolist()
    return predicted_classes
'''
# 遍历每个文件夹
for folder_name in folder_names:
    folder_path = os.path.join(data_path, folder_name)
    output_folder_path = os.path.join(output_path, f"{folder_name}_labeled")
    
    # 创建一个新文件夹来存储处理后的文件
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 遍历文件夹中的CSV文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                
                # 读取CSV文件
                df = pd.read_csv(file_path)
                sentences = df["sentence"].tolist()
                
                # 对句子进行分类并获取标签
                labels = classify_sentences(sentences)
                
                # 将句子转换为张量并移动到CUDA设备上
                inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs = {key: inputs[key].to(device) for key in inputs.keys()}
                
                # 将模型输入和输出移动到CUDA设备上
                model_inputs = {key: inputs[key].to(device) for key in inputs.keys()}
                model_output = torch.tensor(labels).to(device)
                
                # 将标签添加到DataFrame中
                df["label"] = model_output.tolist()
                
                # 将带有标签的DataFrame保存为新的CSV文件
                output_file = os.path.join(output_folder_path, f"labeled_{file}")
                df.to_csv(output_file, index=False)'''
# 遍历每个文件夹
for folder_name in folder_names:
    folder_path = os.path.join(data_path, folder_name)
    output_folder_path = os.path.join(output_path, f"{folder_name}_labeled")

    # 创建一个新文件夹来存储处理后的文件
    os.makedirs(output_folder_path, exist_ok=True)

    # 遍历文件夹中的CSV文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)

                # 读取CSV文件
                df = pd.read_csv(file_path)

                # 分割句子列表为较小的批次
                sentences = df["sentence"].tolist()
                batch_size = 32
                num_batches = (len(sentences) + batch_size - 1) // batch_size

                predicted_classes = []
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(sentences))

                    batch_sentences = sentences[start_idx:end_idx]
                    batch_labels = classify_sentences(batch_sentences)

                    predicted_classes.extend(batch_labels)

                    # 清除缓存的GPU内存
                    torch.cuda.empty_cache()

                # 将标签添加到DataFrame中
                df["label"] = predicted_classes

                # 将带有标签的DataFrame保存为新的CSV文件
                output_file = os.path.join(output_folder_path, f"labeled_{file}")
                df.to_csv(output_file, index=False)