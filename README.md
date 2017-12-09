# Kaggle Competition: Porto Seguro’s Safe Driver Prediction #
This repository contains the code to solve Kaggle contest "Porto Seguro’s Safe Driver Prediction".  
Using **Gradient boosting (XGBoost)** and **Artificial Neural Network (keras)**,  
and **Blending linear combination method**.

### Contributors ###
Chen-Hsi (Sky) Huang (github.com/skyhuang1208)   
Louis Yang (github.com/louis925)  
Luyao Zoe Xu  
Ming Chiu  

### Achievement ###
**Silver medal**, **164th** of 5169 in ranking  

### Workflow ###
1. Train several "XGBoost" and "Neural network" models
In the training stages, some or all of the following techniques were used:
    - Log1p and cubic transform on right-skewed and left-skewed features
    - Cross validation
    - Grid search
2. Blening - linear combination of all predictions from different models
    - Cross validation: determine combination weights
    - Coarse grid & fin random search
    - Probability vs rank: the combination were done using prob. or rank combine

### Scripts ###
- blend1_search.py: input validation predictions, search for best weights w.r.t. Gini coeff. or log loss
- blend2_combine.py: input predictions on test set and weights, output final submission
- blend_rank_1search.py: rank combine version of blend1
- blend_rank_2combine.py: rank combine version of blend2
- model4_XGB_LX_lb280_arxiv.ipynb: train XGBoost model script
- xgboost_sky_buildBlendModel_5_1000.py: train XGBoost model script
- porto_nn_tf.ipynb: train neural network model script
- random_forest_sky_v0.py: train random forest model script
- split_TrainVal.py: split training dataset into train, validation used to determine blend weights


