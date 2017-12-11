from __future__ import print_function
# ## Import Libraries
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import log_loss

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



if len(sys.argv) <5 or not sys.argv[1].isdigit() or int(sys.argv[1])+4 != len(sys.argv):
    print('grid_search')
    exit('Usage: %s _N_models_ [actual] [predict 1] [predict 2] [predict 3] ... [out_file_name]')

print('\nLinear Combination Model (blend). Search coeffs\n')
isgini= input('Choose metrics to be log loss or gini (0: log loss, 1: gini)\n')
isfine= input('Do you want fine tune after grid search? (0: no, 1: yes)\n')

import time
t0= time.time()

Nmds= int(sys.argv[1])

########## READ FILES ##########
actual = pd.read_csv(sys.argv[2], na_values=-1)
y_actual = actual['target'].values
y_pred = []
for i in range(Nmds):
    model = pd.read_csv(sys.argv[i+3])
    y_pred.append(model['target'].values)


### PARS ###
Y_SIZE= y_actual.shape[0]
NUM_RATIO= 100          # N of possible ratio
GRID_SIZE= 1./NUM_RATIO # grid size (period)
N_FINE= 1000
### PARS ###


########## GRID SEARCH ##########
results= {}
coeff= [0 for _ in range(Nmds)]
coeff[-1]= NUM_RATIO
while True:
    # Calculate gini score
    y_blend= np.zeros(Y_SIZE)
    for i, c in enumerate(coeff):
        ratio= c * GRID_SIZE
        y_blend= np.add( ratio * y_pred[i] , y_blend )
    if isgini=='0':
        loss_score = log_loss(y_actual, y_blend)
        results[tuple(np.multiply(coeff, GRID_SIZE))]= loss_score
    else:
        gini_score = gini_normalized(y_actual, y_blend)
        results[tuple(np.multiply(coeff, GRID_SIZE))]= gini_score

    # Make next coeff set
    sumup= sum(coeff[:-1])
    if sumup > NUM_RATIO:
        exit('Err: sum coeff[:-1] too large: %f' % sumup)
    elif sumup == NUM_RATIO:
        if coeff[-2]==NUM_RATIO: break # done searching

        for i, c in enumerate(coeff[:-1]):
            if c != 0:
                coeff[i]= 0
                coeff[i+1] +=1
                if i==len(coeff)-3: print('\r', coeff[-2], '/', NUM_RATIO, end='')
                break
    else: 
        coeff[0] += 1

    coeff[-1]= NUM_RATIO - sum(coeff[:-1])

if isgini=='0': results= sorted(results.items(), key= lambda x:x[1], reverse= False)
else:           results= sorted(results.items(), key= lambda x:x[1], reverse= True)


########## FINE RANDOM SEARCH ##########
if isfine != '0':
    print('\nNow doing random fine-search...\n')

    ##### Generate coeffs randomly deviated from best
    def rand_coeffs(coeffs):
        Nc= len(coeffs)

        sigma= 0.5 * GRID_SIZE 
        def gen_rand():
            rand= sigma * np.random.randn()
            return rand if rand > 0. else 0
        
        while True:
            ipick= np.random.randint(Nc) # pick one not random (force sum to 1.)
            new= np.zeros(Nc)
            total= 0.
            for i, c in enumerate(coeffs):
                if i==ipick: continue
                new[i]= c + gen_rand()
                total += new[i]
            if total <= 1.:
                new[ipick]= 1. - sum(new)
                return new


    ##### Iteratively search for better coeffs #####
    best= list(results[0]) # [0]: coeffs, [1]: score
    results_fine= {}
    for Niter in range(N_FINE): # start random search
        if Niter%10 == 0: print('\r%d/%d' % (Niter, N_FINE), end='')

        # make new coeffs
        new_coeffs= rand_coeffs(best[0])
        y_blend= np.zeros(Y_SIZE)
        for i, c in enumerate(new_coeffs):
            y_blend= np.add( c * y_pred[i], y_blend )
        
        # evaluate
        if isgini=='0':
            loss_score = log_loss(y_actual, y_blend)
            if loss_score < best[1]:
                best= [new_coeffs, loss_score]
                results_fine[tuple(new_coeffs)]= loss_score
        else:
            gini_score = gini_normalized(y_actual, y_blend)
            if gini_score > best[1]:
                best= [new_coeffs, gini_score]
                results_fine[tuple(new_coeffs)]= gini_score
    
    if isgini=='0': results_fine= sorted(results_fine.items(), key= lambda x:x[1], reverse= False)
    else:           results_fine= sorted(results_fine.items(), key= lambda x:x[1], reverse= True)


########## Output results ##########
OFILE= open(sys.argv[-1], 'w')
if isfine != '0':
    print('# Random Fine Tune Results', file=OFILE)
    for key, val in results_fine: 
        for k in key: print('%.5f' % k, file=OFILE, end= ' ')
        print(val, file=OFILE)
    print(file=OFILE)

print('# Grid Search Results', file=OFILE)
for key, val in results:
    print(*key, val, file=OFILE)

print('\nTime spent: ', time.time()-t0)
