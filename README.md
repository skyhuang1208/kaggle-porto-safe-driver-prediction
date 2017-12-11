# Kaggle Competition: Porto Seguro’s Safe Driver Prediction #
A **Blending** approach of **Gradient Boosting (XGBoost)** and **Artificial Neural Network (keras)** to the Kaggle contest - "Porto Seguro’s Safe Driver Prediction".

### Contributors ###
Chen-Hsi (Sky) Huang (https://github.com/skyhuang1208)   
Louis Yang (https://github.com/louis925)  
Luyao Zoe Xu (https://github.com/LuyaoXu)   
Ming-Chang Chiu (https://github.com/charismaticchiu)

### Achievement ###
**Silver medal**, **top 4%** (164th out of 5169 teams) 

### How It Works ###
1. Train several "XGBoost" and "Neural network" models  
    In the training stages, some or all of the following techniques were used:
    - Log1p and cubic transform on right-skewed and left-skewed features
    - Stratified K-fold cross validation
    - Grid search for hyper-parameters
    - Upsampling to enhance the rare positive cases
    - Embedding neural network
2. Blending - Linear combination of all predictions from different models (LCM)
    - Cross validation: determine combination weights
    - Coarse grid search followed by Monte Carlo fine search
    - Probability vs rank blending: combine the result using the predicted probablities or rankings of the probablities

### Scripts ###
- `blend_prob_search.py`: input validation predictions, search for best weights w.r.t. Gini coeff. or log loss
- `blend_prob_combine.py`: input predictions on test set and weights, output final submission
- `blend_rank_search.py`: rank combine version of blend_prob_search
- `blend_rank_combine.py`: rank combine version of blend_prob_combine
- `model_xgboost_sky.py`: XGBoost model 1
- `model_xgboost_luyao.ipynb`: XGBoost model 2
- `model_nn_keras.ipynb`: Keras neural network model
- `model_nn_tf.ipynb`: TensorFlow neural network model (in development)
- `model_random_forest.py`: sklearn random forest model
- `split_train_val.py`: split the training dataset into `LCM_train` and `LCM_val` for blending weight search

### How to Use ###
1. Download and upzip the `train.7z` and `test.7z` from Kaggle (https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data), and place them in the `input` folder.
2. Run `split_train_val.py` to split the training dataset `train.csv` into 3 sets of `LCM_train*.csv` and `LCM_val*.csv` for blending weight search.
3. Train each models on `LCM_train*.csv` and predict on corresponding `LCM_val*.csv` for each LCM set separately. These results are used for determining the blending weights.
4. Train each models on `train.csv` and predict on `test.csv`. These results will be blended together as the final prediction.
5. Run `blend_prob/rank_search.py` to find the best weights using the result on `LCM_val*.csv` set from each model.
6. Run `blend_prob/rank_combine.py` to combine the predictions on `test.csv` from each model with the best weights found in previous step.
