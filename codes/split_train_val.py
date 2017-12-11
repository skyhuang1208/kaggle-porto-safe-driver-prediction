import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train = pd.read_csv('../input/train.csv')
y_train= train['target']

for i in range(3):
	train_, val_ = train_test_split(train, test_size= 0.3, stratify = y_train)

	train_.to_csv('../input/LCM_train'+str(i)+'.csv', index= False, header= True)
	val_.to_csv('../input/LCM_val'+str(i)+'.csv', index= False, header= True)
