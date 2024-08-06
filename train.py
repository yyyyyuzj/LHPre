import os
import argparse
import pandas as pd
import numpy as np
import  torch
from  torch import nn
from  torch.nn import functional as F
from  torch import optim
import  torch.utils.data as Data
from    sklearn.model_selection import KFold
import  pickle as pkl
from model import Transformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from torchsampler import ImbalancedDatasetSampler
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import time 

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


#python train.py --task multi --midfolder ./ --out model --nepoch 150 --trainfile train1 --valfile val1 --trainlabel label_train1 --vallabel label_val1
parser = argparse.ArgumentParser(description="""Main script of LHPre.""")
parser.add_argument('--trainfile', help='patches for training')
parser.add_argument('--valfile', help='patches for validation')
parser.add_argument('--trainlabel', help='patches for training')
parser.add_argument('--vallabel', help='patches for validation')
parser.add_argument('--nepoch', help='number of epoch for training', type=int)
parser.add_argument('--midfolder', help='pth to the midfolder foder', default = './')
parser.add_argument('--out', help='pth to the output foder', default = 'out/')
inputs = parser.parse_args()

start_time = time.time()

mid_fn = inputs.midfolder
out_fn = inputs.out
nepoch = inputs.nepoch
train = inputs.trainfile
val = inputs.valfile
train_label = inputs.trainlabel
val_label = inputs.vallabel


train = pkl.load(open(f"{mid_fn}/{train}", 'rb'))
val = pkl.load(open(f"{mid_fn}/{val}", 'rb'))

train_labels = pkl.load(open(f"{mid_fn}/{train_label}", 'rb'))
val_labels = pkl.load(open(f"{mid_fn}/{val_label}", 'rb'))

#print(converted_train)
print(train.shape)
print(val.shape)
print(train_labels)
print(val_labels)
num_of_class = len(set(train_labels))
num_of_class_test = len(set(val_labels))
print("train_labels", num_of_class)
print("val_labels", num_of_class_test)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_model():
    model = Transformer(
                src_vocab_size=train.shape[2],  # 256
                src_pad_idx=0,
                device=device,
                max_length=train.shape[1],  #
                dropout=0.2,  
                out_dim=num_of_class 
    ).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 0.1、0.01、0.001
    loss_func = nn.CrossEntropyLoss()
    return model, optimizer, loss_func


def return_batch(train_sentence, label, flag, drop):
    X_train = torch.from_numpy(train_sentence).float()
    y_train = torch.from_numpy(label).long()
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,    
        batch_size=64,
        shuffle=flag,               
        num_workers=0,
        drop_last=drop              
    )
    return training_loader


def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


model, optimizer, loss_func = reset_model()
model.to(device)

training_loader = return_batch(train, train_labels, flag=True, drop=False)
val_loader = return_batch(val, val_labels, flag=False, drop=False)
max_f1 = 0
nepoch = tqdm(range(nepoch))
# 指定保存文件路径和名称
file_path = 'dna.txt'

# 逐个 epoch 处理并保存分类报告
with open(file_path, 'w') as f:
    for epoch in nepoch:
        _ = model.train()
        for step, (batch_x, batch_y) in enumerate(training_loader): 
            prediction = model(batch_x.to(device))  # here
            loss = loss_func(prediction.squeeze(1), batch_y.to(device))
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
        _ = model.eval()
        with torch.no_grad():
            all_pred = []
            for step, (batch_x, batch_y) in enumerate(test_loader): 
                logit = model(batch_x.to(device))
                pred = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
                all_pred += pred
            #accuracy = accuracy_score(y_test, y_predlabel)
            
            f1 = accuracy_score(val_labels, all_pred)
            #print('accuracy', f1)
            if max_f1 < f1:
                max_f1 = f1 
                torch.save(model.state_dict(), f'{out_fn}/transformer_multi.pth')
                print("testing:")
                f.write(f"Epoch {epoch}:\n")
                f.write("test\n")
                print(val_labels.shape)
                print(len(all_pred))
                test_report = classification_report(val_labels, all_pred, digits=4)
                print(test_report)
                f.write(test_report)
                f.write("\n")
                all_pred = []
                all_label = []
                for step, (batch_x, batch_y) in enumerate(tqdm(training_loader)):
                    logit = model(batch_x.to(device))
                    pred = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
                    all_pred += pred
                    all_label += batch_y.cpu().detach().numpy().tolist()
                all_label = np.array(all_label).reshape(-1)
                print("training:")
                print(classification_report(all_label, all_pred, digits=4))
                epoch_report = classification_report(all_label, all_pred, digits=4)
                f.write("train\n")
                f.write(epoch_report)
                f.write('\n')


# 记录程序结束时间
end_time = time.time()

# 计算程序运行时间
run_time = end_time - start_time

# 打印程序运行时间
print("程序运行时间：", run_time, "秒")
                
           