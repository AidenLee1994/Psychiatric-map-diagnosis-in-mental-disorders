from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Dataset(Dataset):
    def __init__(self,x,y):
        old_shape=np.shape(x)
        w_h=int(old_shape[1])
        new_shape=(-1,w_h)
        self.x=x.reshape(new_shape)
        self.y=y

    def __getitem__(self, index):
        return self.x[index],self.y[index]

    def __len__(self):
        return len(self.y)


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

    test_data_set=Dataset(np.array(data),np.array(label,dtype=int))
    return train_data_set,test_data_set

