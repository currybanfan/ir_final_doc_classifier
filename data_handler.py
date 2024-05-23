import re
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

class DataHandler:
  def __init__(self, tokenizer_name='bert-base-chinese'):
    self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

  def save_encoded_data(self, directory, encodings, labels):
    with open(f'{directory}/encodings.pickle', 'wb') as handle:
      pickle.dump(encodings, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f'{directory}/labels.pickle', 'wb') as handle:
      pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


  def load_encoded_data(self, directory):
    with open(f'{directory}/encodings.pickle', 'rb') as handle:
      encodings = pickle.load(handle)

    with open(f'{directory}/labels.pickle', 'rb') as handle:
      labels = pickle.load(handle)

    return encodings, labels
  
  def gen_encoded_data(self, texts, max_length, truncation=True, padding=True):
    texts, labels = shuffle(texts, labels, random_state=42)

    encodings = self.tokenizer(texts, truncation=truncation, padding=padding, max_length=max_length)
    
    return encodings

  def create_dataset(self, encodings, labels):
    return TextDataset(encodings, labels)
    
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