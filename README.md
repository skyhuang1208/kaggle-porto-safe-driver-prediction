# Kaggle Competition: Porto Seguro’s Safe Driver Prediction #
This repository contains the code to solve Kaggle contest "Porto Seguro’s Safe Driver Prediction".  
Using **Gradient boosting (XGBoost)** and **Artificial Neural Network (keras)**,  
and **Blending linear combination method**.

### Contributors ###
Chen-Hsi (Sky) Huang (https://github.com/skyhuang1208)   
Louis Yang (https://github.com/louis925)  
Luyao Zoe Xu (https://github.com/LuyaoXu)   
Ming-Chang Chiu (https://github.com/charismaticchiu)

### Achievement ###
**Silver medal**, **Top 4%** (164th out of 5169 teams) 

### Workflow ###
1. Train several "XGBoost" and "Neural network" models
In the training stages, some or all of the following techniques were used:
    - Log1p and cubic transform on right-skewed and left-skewed features
    - Cross validation
    - Grid search
2. Blending - linear combination of all predictions from different models (LCM)
    - Cross validation: determine combination weights
    - Coarse grid & fine random search
    - Probability vs rank combine: the combination was done for the predicted probablities or rankings of the probablities

### Scripts ###
- blend_prob_search.py: input validation predictions, search for best weights w.r.t. Gini coeff. or log loss
- blend_prob_combine.py: input predictions on test set and weights, output final submission
- blend_rank_search.py: rank combine version of blend_prob_search
- blend_rank_combine.py: rank combine version of blend_prob_combine
- model_xgboost_luyao.ipynb: XGBoost model
- model_xgboost_sky.py: XGBoost model
- model_nn_tf.ipynb: TensorFlow neural network model
- model_random_forest.py: sklearn random forest model
- split_train_val.py: split the training dataset into LCM_train and LCM_val for blending weight search
