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
EPOCHS = 6
N_SPLITS = 5

TRAIN_PATH = "Train_Full"
TEST_PATH = "Test_Full"

category2id = {category: idx for idx, category in enumerate(os.listdir(TRAIN_PATH))}
print(category2id)

datanews = []
def read_file(category, file_name):
    text_path = os.path.join(TRAIN_PATH, category, file_name)
    with open(text_path, "r", encoding = 'utf-16') as file:
        content = file.read()
    return (content, category)
#print("TRAIN:")
for category in os.listdir(TRAIN_PATH):
    category_path = os.path.join(TRAIN_PATH, category)
    #print(category_path)
    datanews.extend([ read_file(category, file_name) for file_name in os.listdir(category_path)])
test_data = []
#print("TEST:")
for category in os.listdir(TEST_PATH):
    category_path = os.path.join(TRAIN_PATH, category)
    #print(category_path)
    test_data.extend([ read_file(category, file_name) for file_name in os.listdir(category_path)])

train_data, valid_data = train_test_split(datanews, test_size = 0.1)

train_df = pd.DataFrame(train_data, columns=["text", "label"])
valid_df = pd.DataFrame(valid_data, columns=["text", "label"])
test_df = pd.DataFrame(test_data, columns=["text", "label"])

print(train_df.info())



train_df = pd.concat([train_df, valid_df], ignore_index=True)

skf = StratifiedKFold(n_splits=N_SPLITS)
for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.label)):
    train_df.loc[val_, "kfold"] = fold
print(train_df.info())

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)

class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=120):
        self.df = df
        self.max_len = max_len
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        """
        To customize dataset, inherit from Dataset class and implement
        __len__ & __getitem__
        __getitem__ should return 
            data:
                input_ids
                attention_masks
                text
                targets
        """
        row = self.df.iloc[index]
        text, label = self.get_input_data(row)
        # Encode_plus will:
        # (1) split text into token
        # (2) Add the '[CLS]' and '[SEP]' token to the start and end
        # (3) Truncate/Pad sentence to max length
        # (4) Map token to their IDS
        # (5) Create attention mask
        # (6) Return a dictionary of outputs
        encoding = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_masks': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
        }
    


    def labelencoder(self,text):
        if text=='Van hoa':
            return 0
        elif text=='The gioi':
            return 1
        elif text=='Phap luat':
            return 2
        elif text=='Suc khoe':
            return 3
        elif text=='Kinh doanh':
            return 4
        elif text=='Khoa hoc':
            return 5
        elif text=='Chinh tri Xa hoi':
            return 6
        elif text=='Vi tinh':
            return 7
        elif text=='Doi song':
            return 8
        else:
            return 9
        
    def get_input_data(self, row):
        # Preprocessing: {remove icon, special character, lower}
        text = row['text']
        text = ' '.join(simple_preprocess(text))
        label = self.labelencoder(row['label'])

        return text, label


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
    

def train(model, criterion, optimizer, train_loader):
    model.train()
    losses = []
    correct = 0

    for data in train_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)
        _, pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f'Train Accuracy: {correct.double()/len(train_loader.dataset)} Loss: {np.mean(losses)}')

def eval(test_data = False):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = test_loader if test_data else valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_masks'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)
            correct += torch.sum(pred == targets)
            losses.append(loss.item())
    
    if test_data:
        print(f'Test Accuracy: {correct.double()/len(test_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(test_loader.dataset)
    else:
        print(f'Valid Accuracy: {correct.double()/len(valid_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(valid_loader.dataset)
    
def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = SentimentDataset(df_train, tokenizer, max_len=120)
    valid_dataset = SentimentDataset(df_valid, tokenizer, max_len=120)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    return train_loader, valid_loader

for fold in range(skf.n_splits):
    print(f'-----------Fold: {fold+1} ------------------')
    train_loader, valid_loader = prepare_loaders(train_df, fold=fold)
    model = SentimentClassifier(n_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
    # Batchsize: 16, 32
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    lr_scheduler = get_linear_schedule_with_warmup(
                optimizer, 
                num_warmup_steps=0, 
                num_training_steps=len(train_loader)*EPOCHS
            )
    best_acc = 0
    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-'*30)

        train(model, criterion, optimizer, train_loader)
        val_acc = eval()

        if val_acc > best_acc:
            torch.save(model.state_dict(), f'phobert_fold{fold+1}.pth')
            best_acc = val_acc

def test(data_loader):
    models = []
    for fold in range(skf.n_splits):
        model = SentimentClassifier(n_classes=10)
        model.to(device)
        model.load_state_dict(torch.load(f'phobert_fold{fold+1}.pth'))
        model.eval()
        models.append(model)

    texts = []
    predicts = []
    predict_probs = []
    real_values = []

    for data in data_loader:
        text = data['text']
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        total_outs = []
        for model in models:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                total_outs.append(outputs)
        
        total_outs = torch.stack(total_outs)
        _, pred = torch.max(total_outs.mean(0), dim=1)
        texts.extend(text)
        predicts.extend(pred)
        predict_probs.extend(total_outs.mean(0))
        real_values.extend(targets)
    
    predicts = torch.stack(predicts).cpu()
    predict_probs = torch.stack(predict_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    print(classification_report(real_values, predicts))
    return real_values, predicts

test_dataset = SentimentDataset(test_df, tokenizer, max_len=50)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)
real_values, predicts = test(test_loader)

class_names = ['Van hoa', 'The gioi', 'Phap luat', 'Suc khoe', 'Kinh doanh', 'Khoa hoc', 'Chinh tri Xa hoi', 'Vi tinh', 'Doi song', 'The thao']
def check_wrong(real_values, predicts):
    wrong_arr = []
    wrong_label = []
    for i in range(len(predicts)):
        if predicts[i] != real_values[i]:
            wrong_arr.append(i)
            wrong_label.append(predicts[i])
    return wrong_arr, wrong_label


for i in range(15):
    print('-'*50)
    wrong_arr, wrong_label = check_wrong(real_values, predicts)
    print(test_df.iloc[wrong_arr[i]].text)
    print(f'Predicted: ({class_names[wrong_label[i]]}) --vs-- Real label: ({class_names[real_values[wrong_arr[i]]]})')


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
    attention_mask = encoded_review['attention_mask'].to(device)

    output = model(input_ids, attention_mask)
    _, y_pred = torch.max(output, dim=1)

    print(f'Text: {text}')
    print(f'Sentiment: {class_names[y_pred]}')

infer('Cty Cisco Systems VN đã cho công bố đường dây nóng sử dụng dịch vụ điện thoại miễn phí 1800 tại Việt Nam với ba số điện thoại riêng biệt dành cho hỗ trợ khách hàng, hỗ trợ đại lý và hỗ trợ chung của Trung tâm Hỗ trợ Kỹ thuật Cisco. Đó là các số điện thoại: 1800 585807, 1800 585808, 1800 585809. Đường dây nóng này sẽ hoạt động liên tục 24/24 và 7 ngày trong tuần nhằm giải đáp những thắc mắc liên quan tới thông tin sản phẩm của Cisco, các hoạt động dành cho đại lý và các kênh phân phối cũng như các dịch vụ của Cisco Systems. Tất cả chi phí điện thoại gọi đến những đường dây nóng này từ bất cứ địa điểm nào tại Việt Nam sẽ do Cisco Systems VN thanh toán. Với việc đưa ra dịch vụ hỗ trợ này, Cisco Systems VN đã trở thành một trong những công ty nước ngoài đầu tiên sử dụng dịch vụ điện thoại miễn phí 1800 của Công ty Viễn thông Quốc gia (VTN) để mang lại dịch vụ hỗ trợ tốt nhất cho các khách hàng và đối tác của mình tại Việt Nam. Các dịch vụ hỗ trợ kỹ thuật của Cisco sẽ được cung cấp qua thư điện tử, điện thoại hoặc hệ thống thông tin hỗ trợ trực tuyến tại địa chỉ Cisco.com. Trong các trường hợp khẩn cấp với những vấn đề nghiêm trọng, khách hàng, đối tác, đại lý và các nhà phân phối có thể yêu cầu cung cấp dịch vụ hỗ trợ qua điện thoại.', tokenizer)