import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('../../data/train.csv')
y_train= train['target']
train_, val_ = train_test_split(train, test_size= 0.3, stratify = y_train)

train_.to_csv('dataLCM_train.csv', index= False, header= True)
val_.to_csv('dataLCM_val.csv', index= False, header= True)
