import os
import argparse
import pandas as pd
import numpy as np
import  torch
from    torch import nn
from    torch.nn import functional as F
from    torch import optim
import  torch.utils.data as Data
from    sklearn.model_selection import KFold
import  pickle as pkl
from model import Transformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

parser = argparse.ArgumentParser(description="""Main script of LHPre.""")
parser.add_argument('--file', help='input patches')
parser.add_argument('--labels', help='test labels')
parser.add_argument('--midfolder', help='pth to the midfolder foder', default='midfolder/')
parser.add_argument('--out', help='pth to the output foder', default='out/')
parser.add_argument('--outfile', help='name of the output file', default='final_prediction.csv')
inputs = parser.parse_args()


file_fn = inputs.file 
labels_fn = inputs.labels
mid_fn = inputs.midfolder
out_fn = inputs.out
outfile = inputs.outfile

converted_test = pkl.load(open(f"{mid_fn}/{file_fn}", 'rb'))
test_labels = pkl.load(open(f"{mid_fn}/{labels_fn}", 'rb'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dim = len(set(test_labels))

def reset_model():
    model = Transformer(
                src_vocab_size = converted_test.shape[2],#256
                src_pad_idx = 0,
                device=device,
                max_length=converted_test.shape[1],#16
                dropout=0.1,
                out_dim = out_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    return model, optimizer, loss_func


def return_softmax(all_score):
    result = []
    for item in all_score:
       result.append(softmax(item))
    return np.array(result)


def return_batch(train_sentence, label, flag, drop):
    X_train = torch.from_numpy(train_sentence).float()
    y_train = torch.from_numpy(label).long()
    train_dataset = Data.TensorDataset(X_train, y_train)
    training_loader = Data.DataLoader(
        dataset=train_dataset,    
        batch_size=16,
        shuffle=flag,               
        num_workers=0,
        drop_last=drop              
    )
    return training_loader


def return_tensor(var, device):
    return torch.from_numpy(var).to(device)


model, optimizer, loss_func = reset_model()

pretrained_dict = torch.load('/.../model/transformer_multi.pth', map_location=device)
model.load_state_dict(pretrained_dict)

test_loader = return_batch(converted_test, test_labels, flag=False, drop=False)
model = model.eval()
with torch.no_grad():
    all_pred = []
    all_score = []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        logit = model(batch_x.to(device))
        pred = np.argmax(logit.squeeze(1).cpu().detach().numpy(), axis=1).tolist()
        all_pred += pred
        pred = logit.squeeze(1).cpu().detach().numpy()
        all_score.append(pred)
        
    df = pd.DataFrame()
    df['preds'] = all_pred
    df['labels'] = test_labels
    df.to_csv('test_res.csv', index=False)
    print(classification_report(test_labels, all_pred, digits=4))
