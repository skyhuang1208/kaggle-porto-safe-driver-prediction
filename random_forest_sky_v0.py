# coding: utf-8
# # XGBoost model to predict a driver's probablity of filing an insurance claim
#
# The data is taken from Kaggle Competition - Porto Seguro https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

# Modify codes by Sky from LX

# ## Import Libraries
import pandas as pd
import numpy as np
import xgboost as xgb

#import matplotlib.pylab as plt
#get_ipython().magic(u'matplotlib inline')
#from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 12, 4




########## **********  Various functions which I didn't do look  ********** ##########


# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred):
    assert (len(actual) == len(pred))
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):
    return gini(actual, pred) / gini(actual, actual)

# Create an XGBoost-compatible Gini metric
def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', gini_score)]

def gini_xgb_min(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return [('gini', -1*gini_score)]

# Some functions from Andy: https://www.kaggle.com/aharless/xgboost-cv-lb-284



########## **********  Various functions which I didn't do look   ********** ##########





# Data exploration
train = pd.read_csv('data/train.csv', na_values=-1)
if True: # if pars search, turn off can reduce mem use
    test = pd.read_csv('data/test.csv', na_values=-1)

pos_count = train[train['target']==1]['target'].count() 
neg_count = train.shape[0] - pos_count
pos_count_ratio = float(pos_count)/train.shape[0]
print('N of train (all, pos, neg, ratio)', \
      train.shape[0], pos_count, neg_count, pos_count_ratio)


# Group different types of data
feats_ordi = []
feats_cat = []
feats_bin = []
for feat in train.columns:
    if '_cat' in feat:      feats_cat.append(feat)
    elif '_bin' in feat:    feats_bin.append(feat)
    elif feat != 'id' and feat != 'target':
                            feats_ordi.append(feat)

print('Number of ordinal features: ', len(feats_ordi))
print('Number of categorical features: ', len(feats_cat))
print('Number of binary features: ', len(feats_bin))


#Define indice for features (i.e. predictors) and labels (i.e. target)
target ='target'
IDcol = 'id'
features = [x for x in train.columns if x not in [target, IDcol]]

from scipy.stats import skew # check if log transform is necessary
skewed_feats = train[feats_ordi].apply(lambda x: skew(x.dropna())) #compute skewness
print('Skewness:')
print(skewed_feats)
if True: # perform transform to deal with skewness (seems not good)
    right_skewed = skewed_feats[skewed_feats > 0.5].index
    left_skewed  = skewed_feats[skewed_feats < -0.5].index
    train[right_skewed] = np.log1p(train[right_skewed])
    train[left_skewed]  = np.power(train[left_skewed], 3)
    test[right_skewed] = np.log1p(test[right_skewed])
    test[left_skewed]  = np.power(test[left_skewed], 3)

X_train = train[features]
y_train = train[target]
X_test = test[features]


#Create dummies for categorical features

#X_train[feats_cat] = X_train[feats_cat].fillna(-1) #  Seems xgb will deal with nan 

if False: # Drop 11 'cause it has many cats (can reduce iter runs)
    X_train.drop(['ps_car_11_cat'], axis=1, inplace=True)
#    X_test.drop(['ps_car_11_cat'], axis=1, inplace=True)
    feats_cat.remove('ps_car_11_cat')

#print(X_train.columns)
X_train[feats_cat] = X_train[feats_cat].astype(object)
X_train = pd.get_dummies(X_train, dummy_na= True)
#print(X_train.columns)

#print(X_test.columns)
X_test[feats_cat] = X_test[feats_cat].astype(object)
X_test = pd.get_dummies(X_test, dummy_na= True)
#print(X_test.columns)


# fillna
from sklearn.preprocessing import Imputer
fillnan= Imputer()
X_train= fillnan.fit_transform(X_train)
fillnan= Imputer()
X_test=  fillnan.fit_transform(X_test)



##############################     PARS SEARCH     ##############################
def gridSearch(X, y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer
    from sklearn.preprocessing import Imputer
    from sklearn.model_selection import ShuffleSplit
    from numpy.random import randint
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # X, y size
    # X, y size
    X_dump, X_, y_dump, y_ = train_test_split(X, y, test_size= 100000, stratify = y)

    # Define RF
    RFclf= RandomForestClassifier()

    # Parameters 
    parameters={'max_depth':            [3,7,10,20,None],
                'min_samples_split':    [2, 5, 10], 
                'class_weight':         [{0:1, 1:1}, {0:1, 1:2}, {0:1, 1:10}] }
    
    # Greeting #
    print('%%% Start to do grid search %%%')
    print('Searching parameters:')
    print([ p for p in parameters.items() if len(p[1]) != 1 ])

    # Define shuffle set
    cv_sets = ShuffleSplit(n_splits = 5, test_size = 0.2) # Rand sets instead of N-fold

    # Run gridSearch
    gridRF = GridSearchCV(estimator= RFclf, scoring= 'neg_log_loss', param_grid= parameters, cv= cv_sets)
    gridRF.fit(X_, y_)
   
    results= pd.DataFrame(gridRF.cv_results_).loc(axis=1)['params', 'rank_test_score', 'mean_test_score', 'mean_train_score']
    print(results)



if False: # gridSearch using sklearn gridsearchcv
    gridSearch(X_train, y_train.values)


##############################     PARS SEARCH     ##############################
    


def train_pred(Xtrain, ytrain, Xtest):
    from sklearn.ensemble import RandomForestClassifier
    
    RFclf= RandomForestClassifier(max_depth= 50, min_samples_split= 500, n_estimators= 1000)
    
    RFclf.fit(Xtrain, ytrain)
    
    # Predict test set:
    return RFclf.predict_proba(Xtest)[:,1]


if True: # Train & Predict
    y_pred= train_pred(X_train, y_train, X_test)
        
    # Save test results to csv file
    pd.DataFrame({"id": test['id'], "target": y_pred}).to_csv('sub_sky_rf_maxd10minsample10.csv', index=False, header=True)
