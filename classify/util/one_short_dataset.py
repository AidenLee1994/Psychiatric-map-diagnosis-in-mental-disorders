from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import  Counter
import random

class Dataset(Dataset):
    def __init__(self,x,y):
        old_shape=np.shape(x)
        w_h=int(np.sqrt(old_shape[1]))
        new_shape=(-1,1,w_h,w_h)
        self.x=x.reshape(new_shape)
        self.y=y

    def __getitem__(self, index):
        flags=[i for i in range(len(self.y))]
        pic_1_flags=random.choice(flags)
        if index % 2==1:
            label=1
            while True:
                pic_2_flags=random.choice(flags)
                if self.y[pic_1_flags]==self.y[pic_2_flags]:
                    img1=self.x[pic_1_flags]
                    img2=self.x[pic_2_flags]
                    break
        else:
            label=0
            while True:
                pic_2_flags=random.choice(flags)
                if self.y[pic_1_flags]!=self.y[pic_2_flags]:
                    img1=self.x[pic_1_flags]
                    img2=self.x[pic_2_flags]
                    break
        return (img1,img2,label)


    def __len__(self):
        classes=Counter(self.y)
        num=0
        for i in classes.keys():
            for j in classes.keys():
                if i!=j:
                    num=classes[i]*classes[j]
                else:
                    continue
        return num

def get_data(percent=0.1):
    data=pd.read_csv("../data/"+str(percent)+"_prototype.csv",header=1)
    columns=data.columns
    label=[]
    for item in columns:
        info=item.split("_")
        if info[0]=="control":
            label.append(0)
        elif info[1]=="BPD":
            label.append(1)
        else:
            label.append(2)
    data=np.array(data).T
    X_train, X_test, y_train, y_test = train_test_split(np.array(data), np.array(label,dtype=int), test_size = 0.3, random_state = 52013)


    train_data_set=Dataset(X_train,y_train)

    test_data_set=Dataset(X_test,y_test)
    return train_data_set,test_data_set

