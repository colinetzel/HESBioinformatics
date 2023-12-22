import deepRAM
import argparse
import gzip
import csv
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn import metrics

from torch.utils.data import Dataset

def main():
    parser = argparse.ArgumentParser(description='sequence specificities prediction using random forest and SVM approach')
    args = parse_arguments(parser)
    train_data = args.train_data
    test_data = args.test_data
    data_type = args.data_type
    chipseq = Chip(train_data)
    alldataset=chipseq.openFile()
    alldataset_dataset=deepRAM.chipseq_dataset(alldataset)
    train1_numpy_x = alldataset_dataset.x_data.numpy()
    train1_numpy_y = alldataset_dataset.y_data.numpy()
    train2d_numpy_y = train1_numpy_y.reshape(-1)
    train2d_numpy_x = train1_numpy_x.reshape(train1_numpy_x.shape[0], -1)

    clf = svm.SVC(kernel = "poly", degree = 2, probability = True)
    clf.fit(train2d_numpy_x, train2d_numpy_y)

    clf2 = RandomForestClassifier(max_depth=2, random_state=0)
    clf2.fit(train2d_numpy_x, train2d_numpy_y)

    #load test data and evaluate performance
    chipseq2 = Chip(test_data)
    alldataset_2=chipseq2.openFile()
    alldataset_dataset_2=deepRAM.chipseq_dataset(alldataset_2)
    test_numpy_x = alldataset_dataset_2.x_data.numpy()
    test_numpy_y = alldataset_dataset_2.y_data.numpy()

    test2d_numpy_y = test_numpy_y.reshape(-1)
    test2d_numpy_x = test_numpy_x.reshape(test_numpy_x.shape[0], -1)

    print("SVM - validation R2 score")
    print(clf.score(train2d_numpy_x,train1_numpy_y))

    print("SVM - test R2 score")
    print(clf.score(test2d_numpy_x, test2d_numpy_y))
    print("SVM - roc/auc score")
    print(metrics.roc_auc_score(test2d_numpy_y, clf.decision_function(test2d_numpy_x)))


    print("Random Forest - validation R2 score")
    print(clf2.score(train2d_numpy_x,train1_numpy_y))

    print("Random Forest - test R2 score")
    print(clf2.score(test2d_numpy_x, test2d_numpy_y))
    print("Random Forest - roc/auc score")
    print(metrics.roc_auc_score(test2d_numpy_y, clf2.predict_proba(test2d_numpy_x)[:,1]))

def parse_arguments(parser):
## data
    parser.add_argument('--train_data', type=str, default='train_ChIP.gz',
                        help='path for training data with format: sequence 	label')
    
    parser.add_argument('--test_data', type=str, default='test_ChIP.gz',
                        help='path for test data containing test sequences with or without label')
    parser.add_argument('--data_type', type=str, default='RNA',
                        help='type of data: DNA or RNA ')

    
    args = parser.parse_args()

    return args

#slight modification to Chip class in deepRAM
class Chip():
    def __init__(self,filename,motiflen=24):
        self.file = filename
        self.motiflen = motiflen
  
    def openFile(self):
        train_dataset=[]
        sequences=[]
        with gzip.open(self.file, 'rt') as data:
                next(data)
                reader = csv.reader(data,delimiter='\t')

                for row in reader:           
                    train_dataset.append([seqtopad(row[0],self.motiflen),[int(row[1])]])
  
        return train_dataset

#slight modification to seqtopad class in deepRAM
def seqtopad(sequence,motlen):
    rows=len(sequence)+2*motlen-2
    S=np.empty([rows,4])
    base= "ACGU" #rna bases
    for i in range(rows):
        for j in range(4):
            if i-motlen+1<len(sequence) and sequence[i-motlen+1]=='N' or i<motlen-1 or i>len(sequence)+motlen-2:
                S[i,j]=np.float32(0.25)
            elif sequence[i-motlen+1]==base[j]:
                S[i,j]=np.float32(1)
            else:
                S[i,j]=np.float32(0)
    return np.transpose(S)

#slight modification to chipseq_dataset class in deepRAM
class chipseq_dataset(Dataset):

    def __init__(self,xy=None):
        print(xy)
        self.x_data=np.asarray([el[0] for el in xy],dtype=np.float32)
        self.y_data =np.asarray([el[1] for el in xy ],dtype=np.float32)
        self.len=len(self.x_data)
      

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

#call main function
main()