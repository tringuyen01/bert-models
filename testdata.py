import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm_notebook as tqdm
from sklearn.model_selection import train_test_split

from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup, AutoModel, logging
import warnings

warnings.filterwarnings("ignore")

logging.set_verbosity_error()
torch.cuda.memory_summary(device=None, abbreviated=False)

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(86)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained("vinai/phobert-base")
        self.drop = nn.Dropout(p=0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False # Dropout will errors if without this
        )

        x = self.drop(output)
        x = self.fc(x)
        return x
    

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
model = SentimentClassifier(n_classes=10).to(device)
model.load_state_dict(torch.load('phobert_fold1.pth'))
model.load_state_dict(torch.load('phobert_fold2.pth'))
model.load_state_dict(torch.load('phobert_fold3.pth'))
model.load_state_dict(torch.load('phobert_fold4.pth'))
model.load_state_dict(torch.load('phobert_fold5.pth'))


class_names = ['Van hoa', 'The gioi', 'Phap luat', 'Suc khoe', 'Kinh doanh', 'Khoa hoc', 'Chinh tri Xa hoi', 'Vi tinh', 'Doi song', 'The thao']

def infer(text, tokenizer, max_len=120):
    encoded_review = tokenizer.encode_plus(
        text,
        max_length=max_len,
        truncation=True,
        add_special_tokens=True,
        padding='max_length',
        return_attention_mask=True,
        return_token_type_ids=False,
        return_tensors='pt',
    )

    input_ids = encoded_review['input_ids'].to(device)
    print(input_ids)
    attention_mask = encoded_review['attention_mask'].to(device)    
    print(attention_mask)
    output = model(input_ids, attention_mask)
    print(output)
    _, y_pred = torch.max(output, dim=1)
    
    print(y_pred)
    print(f'Text: {text}')
    print(f'Prediction: {class_names[y_pred]}')

infer('U20 VN vào bán kết Thắng giòn giã Maldives 4-0 trong trận cuối cùng của vòng loại bảng B, đội tuyển bóng đá U20 VN giành một suất vào bán kết giải vô địch U20 Đông Nam Á, với 8 điểm sau 4 trận.Đối thủ của U20 VN nhiều khả năng là đội U20 Myanmar, đội toàn thắng trong 3 trận vòng loại ở bảng A và còn một trận chưa đấu với Indonesia. Do bảng A hôm nay, 13-8, mới kết thúc nên U20 VN có lợi thế hơn đối thủ ở bán kết 1 ngày nghỉ. ', tokenizer)