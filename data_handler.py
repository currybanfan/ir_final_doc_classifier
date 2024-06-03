import pickle
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

def save_encoded_data(directory, data):
  file_path = f'{directory}.pickle'
  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  with open(file_path, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_encoded_data(directory):
  with open(f'{directory}.pickle', 'rb') as handle:
    data = pickle.load(handle)

  return data

def clean_text(text):
    text = re.sub(r'※ \[.*?\] ', '', text)
    text = re.sub(r'作者: .*? \(.*?\) 看板: .*? 標題: .*? 時間:\s+\w{3}\s+\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4}', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'^※.*?之銘言：$', '', text, flags=re.MULTILINE)
    text = re.sub(r'.{0,5}?(網誌：|網誌版：|圖文：|圖文版：)', '', text)
    text = re.sub(r'※ 發信站:.*', '', text, flags=re.DOTALL)
    text = re.sub(r'[^\w。，!?"]', '', text, flags=re.UNICODE)
    text = re.sub(r'_+', '', text)

    return text

def create_dataset(encodings, labels=None):
    if labels is None:
       return PredictionTextDataset(encodings)
    else:
      return TextDataset(encodings, labels)

class DataHandler:
  def __init__(self, tokenizer_name='bert-base-chinese'):
    self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
  
  def gen_encoded_data(self, texts, max_length, truncation=True, padding=True):
    encodings = self.tokenizer(texts, truncation=truncation, padding=padding, max_length=max_length)
    
    return encodings
  
  def prepare_input_for_prediction(self, text, max_length=512, truncation=True, padding=True):
    cleaned_text = clean_text(text)
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
    
class PredictionTextDataset(Dataset):
  def __init__(self, encodings):
    self.encodings = {key: torch.tensor(val) for key, val in encodings.items()}

  def __getitem__(self, idx):
    item = {key: val[idx] for key, val in self.encodings.items()}
    return item

  def __len__(self):
    return len(next(iter(self.encodings.values())))