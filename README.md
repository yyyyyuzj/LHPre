**Phage host prediction pipeline based on lytic enzyme information**

**Overview**

**Required Dependencies**

Detailed package information can be found in requirements.txt
If you want to use the gpu to accelerate the program please install the packages below:
cuda
Pytorch-gpu
Search pytorch to find the correct cuda version based on your computer.

**Quick install**

Note: we suggest you to install all the package using conda (both miniconda and Anaconda are ok).
After cloning this respository, you can use anaconda to install the requirements.txt. This will install all packages you need with cpu mode. The command is:pip install -r requirements.txt
Once installed, you only need to activate your 'lhpre' environment before using lhpre in the next time.
conda activate lhpre

**Usage**

1.data_embeddings.py:Perform FCGR encoding and class imbalance handling

2.train data:

python train.py --task multi --midfolder ./ --out model --nepoch 150 --trainfile train1 --valfile val1 --trainlabel label_train1 --vallabel label_val1

trainfile:train data

testfile:test data

trainlabel:train label

testlabel:test label

3.predict data:

python predict.py --file test --labels label_test




