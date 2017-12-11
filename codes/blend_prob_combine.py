import sys


if len(sys.argv) < 3 or not sys.argv[1].isdigit() or int(sys.argv[1])+2 != len(sys.argv):
    print('\nMake blend submissions')
    exit('Usage: %s N_models [test pred 1] [test pred 2] ...')
    
Nmds= int(sys.argv[1])

import pandas as pd
import numpy as np

coeff= []
y_pred = []
for i in range(Nmds):
    # Read in coeff
    c= input('Enter the coefficient of model %d\n' % (i+1))
    coeff.append(float(c))

    # Read csv file
    model = pd.read_csv(sys.argv[i+2])
    if i==0: ids= model['id'].values
    y_pred.append(model['target'].values)
   
if abs(sum(coeff)-1.)>0.0001: print('I do hope the sum of coeffs is 1., but if you insist...')

ofile_name= input('Enter output name (dafault: ../results/blend_prob.csv)\n')
if ofile_name=='': ofile_name= '../results/blend_prob.csv'

y_blend= np.zeros(y_pred[0].shape[0])
for i in range(Nmds):
    y_blend= np.add( coeff[i] * y_pred[i] , y_blend )
pd.DataFrame({'id': ids, 'target': y_blend}).to_csv(ofile_name, index=False, header=True)

print('\nMaking completed. Check < %s >.' % ofile_name)
