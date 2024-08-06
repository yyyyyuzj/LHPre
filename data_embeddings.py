import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader
import  pickle as pkl
from beta_vae import BetaVAE
from fasta2CGR import *
import time


def return_img(feature):
    res = 64
    converted_figure = []
    for item in feature:
        item = np.array(list(item))
        fig = item.reshape(res, res)
        patch_figure = []
        for i in range(0, res, 16):
            for j in range(0, res, 16):
                patch = fig[i:i+16, j:j+16]
                patch = patch.reshape(-1)
                assert len(patch) == 256
                patch_figure.append(patch)
        patch_figure = np.array(patch_figure)
        converted_figure.append(patch_figure)
    converted_figure = np.array(converted_figure)
    return converted_figure


#normalaize g
def normalize(feature):
    norm_feature = []
    for item in feature:
        max_ = np.max(item)
        min_ = np.min(item)
        norm_feature.append((item-min_)/(max_-min_))
    return np.array(norm_feature) 


def calculate_6mer_features(sequence):
    # alphabet
    k_list = ["A", "C", "G", "T"]
    nucl_list = ["A", "C", "G", "T"]
    for i in range(5):
        tmp = []
        for item in nucl_list:
            for nucl in k_list:
                tmp.append(nucl + item)
        k_list = tmp
    # dictionary
    mer2dict = {mer: idx for idx, mer in enumerate(k_list)}
    # convert sequence to 6-mer features
    feature = np.zeros(4096)
    for pos in range(len(sequence) - 5):
        try:
            feature[mer2dict[sequence[pos:pos+6]]] += 1
        except KeyError:
            pass
    # normalization
    norm_feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
    return norm_feature


def ob_vecs(df):
    kmer = 6
    vecs = []
    for feature in df['dna']:
    #print(feature.type)
        seq = feature.replace('N', '')
        fc = count_kmers(seq, kmer)
        f_prob = probabilities(seq, fc, kmer)
        chaos_k = chaos_game_representation(f_prob, kmer)
        vecs.append(chaos_k)
    return vecs

start_time = time.time()

df = pd.read_csv('/your path/dataset.csv')
vecs = ob_vecs(df)
feature = np.array(vecs)
feature = np.reshape(feature, (-1, 4096))
feature = feature.tolist()
label = df['Host_Species'].values

""" # label2int
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(label)
print(len(labels))
df['label2int'] = labels
df.to_csv('dataset.csv', index=False) """

label = df['label2int'].tolist()
counts = df['label2int'].value_counts()
trimmed_counts = counts.nsmallest(len(counts)-1).nlargest(len(counts)-2)
average = trimmed_counts.mean() 
print(average)  # 258
filtered_elements = counts[counts < average].index.tolist()
print(len(filtered_elements))
for element in filtered_elements:
    filtered_df = df[df['label2int'] == element]
    X = ob_vecs(filtered_df)  
    X = np.array(X)
    print(X.shape)
    tensor = torch.from_numpy(X)
    X = torch.unsqueeze(tensor, dim=1)
    print(X.shape)
    model = BetaVAE(in_channels=1, latent_dim=64)
    X = X.to(torch.float)
    output, _, mu, log_var = model.forward(X)
    output = output.squeeze(1)
    print(output.shape)
    output = output.detach().numpy()
    print(output.shape)
    output = output.reshape((-1, 4096))
    print(output.shape)
    label1 = [element] * output.shape[0]
    label.extend(label1)
    feature.extend(output.tolist())


end_time = time.time()
run_time = end_time - start_time
print("run time:", run_time, "秒")

print("Divide the dataset")
label = np.array(label)
blended_vector = np.array(feature)
print(blended_vector.shape)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


for fold, (train_index, val_index) in enumerate(skf.split(blended_vector, label), 1):
    X_train, X_val = blended_vector[train_index], blended_vector[val_index]
    y_train, y_val = label[train_index], label[val_index]
    classes = np.unique(y_train)
    print("classes", len(classes))
    img_train = return_img(X_train)
    converted_train = normalize(img_train)
    img_test = return_img(X_val)
    converted_val = normalize(img_test)
    print(converted_train.shape)

    pkl.dump(converted_train, open(f"/your path/train{fold}", "wb"))
    pkl.dump(converted_val, open(f"/your path/val{fold}", "wb"))
    pkl.dump(y_train, open(f'/your path/label_train{fold}', 'wb'))
    pkl.dump(y_val, open(f'/your path/label_val{fold}', 'wb'))


