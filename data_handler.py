import pickle
from sklearn.utils import shuffle
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import re
import os
import json

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
      data = json.load(file)      
    return data

def save_json(data, file_path):
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

def save_encoded_data(directory, encodings, labels):
  file_path = f'{directory}/encodings.pickle'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'wb') as handle:
    pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)

  file_path = f'{directory}/labels.pickle'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_encoded_data(directory):
  with open(f'{directory}/encodings.pickle', 'rb') as handle:
    encodings = pickle.load(handle)

  with open(f'{directory}/labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)

  return encodings, labels

def clean_texts(texts):
    cleaned_texts = []
    for text in texts:
      text = re.sub(r'※ \[.*?\] ', '', text)
      text = re.sub(r'作者: .*? \(.*?\) 看板: .*? 標題: .*? 時間:\s+\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}', '', text)
      text = re.sub(r'http[s]?://\S+', '', text)
      text = re.sub(r'^※.*?之銘言：$', '', text, flags=re.MULTILINE)
      text = re.sub(r'.{0,5}?(網誌：|網誌版：|圖文：|圖文版：)', '', text)
      text = re.sub(r'※ 發信站:.*', '', text, flags=re.DOTALL)
      text = re.sub(r'[^\w。，!?"]', '', text, flags=re.UNICODE)
      text = re.sub(r'_+', '', text)
      cleaned_texts.append(text)

    return cleaned_texts


class DataHandler:
  def __init__(self, tokenizer_name='bert-base-chinese'):
    self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
  
  def gen_encoded_data(self, texts, labels, max_length, truncation=True, padding=True):
    texts, labels = shuffle(texts, labels, random_state=42)

    encodings = self.tokenizer(texts, truncation=truncation, padding=padding, max_length=max_length)
    
    return encodings

  def create_dataset(self, encodings, labels):
    return TextDataset(encodings, labels)
  
  def prepare_input_for_prediction(self, text, max_length=512, truncation=True, padding=True):
    cleaned_text = clean_texts(text)
    inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=truncation, padding=padding, max_length=max_length)
    return inputs
    
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}  # 提前轉換成 tensor
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}  # 直接索引已有 tensor
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)