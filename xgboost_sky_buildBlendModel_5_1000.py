# coding: utf-8
# # XGBoost model to predict a driver's probablity of filing an insurance claim
#
# The data is taken from Kaggle Competition - Porto Seguro https://www.kaggle.com/c/porto-seguro-safe-driver-prediction

# Modify codes by Sky from LX

# ## Import Libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

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



import sys
if len(sys.argv) != 4:
    exit('Usage %s [train_file](IN) [valid_file](IN) [pred_valid](OUT)')


# Data exploration
train = pd.read_csv(sys.argv[1], na_values=-1)
valid = pd.read_csv(sys.argv[2], na_values=-1)

train_pos= train[train['target']==1]['target'].count()
train_neg= train.shape[0]-train_pos
train_ratio = float(train_pos)/train.shape[0]
valid_pos= valid[valid['target']==1]['target'].count()
valid_neg= valid.shape[0]-valid_pos
valid_ratio = float(valid_pos)/valid.shape[0]
print('\n All Pos Neg Pos_ratio')
print('Train:', train.shape[0], train_pos, train_neg, train_ratio)
print('Valid:', valid.shape[0], valid_pos, valid_neg, valid_ratio)


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

target ='target'
IDcol = 'id'
features = [x for x in train.columns if x not in [target, IDcol]]


# Check skewness
from scipy.stats import skew # check if log transform is necessary
skewed_feats = train[feats_ordi].apply(lambda x: skew(x.dropna())) #compute skewness
print('Skewness:')
print(skewed_feats)
if True: # perform transform to deal with skewness (seems not good)
    right_skewed = skewed_feats[skewed_feats > 0.5].index
    left_skewed  = skewed_feats[skewed_feats < -0.5].index
    train[right_skewed] = np.log1p(train[right_skewed])
    train[left_skewed]  = np.power(train[left_skewed], 3)
    valid[right_skewed] = np.log1p(valid[right_skewed])
    valid[left_skewed]  = np.power(valid[left_skewed], 3)


# Make dataframes
X_train = train[features]
y_train = train[target]
X_valid = valid[features]
y_valid = valid[target]


# Create dummies for categorical features

#X_train[feats_cat] = X_train[feats_cat].fillna(-1) #  Seems xgb will deal with nan 
if False: # Drop 11 'cause it has many cats (can reduce iter runs)
    X_train.drop(['ps_car_11_cat'], axis=1, inplace=True)
    X_test.drop(['ps_car_11_cat'], axis=1, inplace=True)
    feats_cat.remove('ps_car_11_cat')

X_train[feats_cat] = X_train[feats_cat].astype(object)
X_train = pd.get_dummies(X_train, dummy_na= True)
X_valid[feats_cat] = X_valid[feats_cat].astype(object)
X_valid = pd.get_dummies(X_valid, dummy_na= True)


##############################     ACTUAL PREDICT     ##############################

def train_pred(Xtrain, ytrain, Xvalid, isfindSTOP= False):
    from sklearn.model_selection import train_test_split
    
    print('%%%%%%%%%%%%%%%%%%%% Start train model. %%%%%%%%%%%%%%%%%%%%')

    ### FIXED PARS ###
    learn_rate= 0.07
    n_trees= 1000       # previous tested using early stop
    if isfindSTOP: n_trees= 1000
    ### FIXED PARS ###
    
    par_max_depth=          7
    par_gamma=              20
    par_min_child_weight=   10
    par_reg_alpha=          0.0
    par_reg_lambda=         2.0
    par_scale_pos_weight=   1.3
    
    # Define XGBoost classifier
    xgb = XGBClassifier( objective= 'binary:logistic',  seed= np.random.randint(0, 1000000),
                         learning_rate= learn_rate,     n_estimators= n_trees,
                         subsample=0.8,                 colsample_bytree=0.8,
                         max_depth= par_max_depth,      min_child_weight= par_min_child_weight,
                         gamma= par_gamma,  reg_alpha= par_reg_alpha,   reg_lambda= par_reg_lambda,
                         scale_pos_weight= par_scale_pos_weight)

    if isfindSTOP:
        # Use stratified train_test_split due to the very imbalanced label classes
        X_train, X_val, y_train, y_val = train_test_split(Xtrain, ytrain, test_size= 0.1, stratify = ytrain)
        eval_set=[(X_train, y_train), (X_val, y_val)]
        
        # Fit the classifier instance on the training data
        xgb.fit(X_train, y_train, eval_set= eval_set,
            early_stopping_rounds= 50, eval_metric= gini_xgb_min)

        # Predict training set:
        train_predprob = xgb.predict_proba(X_train)[:,1]
        val_predprob = xgb.predict_proba(X_val)[:,1]
        gini_train= gini_normalized(y_train, train_predprob)
        gini_val= gini_normalized(y_val, val_predprob)
        print("Val, Train Gini coef : %.5f %.5f" % (gini_val, gini_train) ) 

    else:
        # Fit the classifier instance on the training data
        xgb.fit(Xtrain, ytrain)
    
        # Predict test sets:
        p_valid= xgb.predict_proba(Xvalid)[:,1]

        return p_valid
   
if False: # Find stopping point
    train_pred(X_train, y_train, X_valid, True)

if True: # Actual train and predict
    valid['pred']= train_pred(X_train, y_train, X_valid)
        
    # Save test results to csv file
    pd.DataFrame({"id": valid['id'], "target": valid['pred']}).to_csv(sys.argv[3], index=False, header=True)



########## %%%%%%%%%%%%% Candidate Parameters %%%%%%%%%%%%%%%% ##############
'''
### Par 1:   LB score: 0.279
no Drop; no TRANS
learn_rate= 0.07
n_trees= 218        # previous tested using early stop
(learn_rate= 0.05, n_trees= 1000 has same result) 
            %%%
par_max_depth=          7
par_gamma=              10
par_min_child_weight=   6
par_reg_alpha=          1
par_reg_lambda=         1.3
par_scale_pos_weight=   1.0

### Par 2:  LB score: 0.278 
no Drop; yes TRANS
learn_rate= 0.07
n_trees= 313
            %%%
par_max_depth=          5
par_gamma=              20
par_min_child_weight=   10
par_reg_alpha=          0.5
par_reg_lambda=         0.0
par_scale_pos_weight=   1.6

### Par 3:
no Drop; no TRANS
learn_rate= 0.07
n_trees=
            %%%
par_max_depth=          3
par_gamma=              1
par_min_child_weight=   1
par_reg_alpha=          0.5
par_reg_lambda=         0.5
par_scale_pos_weight=   1.0

### Par 4:  LB 0.279
no Drop; noTRANS
learn_rate= 0.07
n_trees= 256 
        %%%
par_max_depth=          4
par_gamma=              10
par_min_child_weight=   6
par_reg_alpha=          8  
par_reg_lambda=         1.3
par_scale_pos_weight=   1.6

# Par 5:    LB score 0.280 (0.275 for n_est= 204)
no Drop; yes TRANS
learn_rate= 0.07
    %%%
par_max_depth=          7
par_gamma=              20
par_min_child_weight=   10
par_reg_alpha=          0.0
par_reg_lambda=         2.0
par_scale_pos_weight=   1.3
'''
