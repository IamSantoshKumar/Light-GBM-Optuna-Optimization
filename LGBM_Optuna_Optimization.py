import os
import argparse
import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
import lightgbm as lgbm
import enum
import math
import glob
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def optimize(trial,x,y): 
    lambda_l1 = trial.suggest_float('lambda_l1', 1E-12, 20, log=True)
    lambda_l2 = trial.suggest_float('lambda_l2', 1E-12, 20, log=True)
    bagging_freq = trial.suggest_int('bagging_freq', 1, 100)
    bagging_fraction = trial.suggest_float('bagging_fraction ', 0, 1.0)
    feature_fraction = trial.suggest_float('feature_fraction', 0, 1.0)
    num_leaves = trial.suggest_int('num_leaves', 10, 500)
    min_data_in_leaf = trial.suggest_int('min_data_in_leaf ', 1, 100)
    learning_rate=trial.suggest_loguniform('learning_rate',0.01, 1.0)
    
    lgb_params= {
               'learning_rate': learning_rate,
               'lambda_l1':lambda_l1,
               'lambda_l2':lambda_l2,
               'bagging_freq':bagging_freq,
               'bagging_fraction':bagging_fraction,
               'feature_fraction' : feature_fraction,
               'num_leaves' : num_leaves,
               'min_data_in_leaf' : min_data_in_leaf,
               'n_jobs': -1
    }
    
    num_rounds=args.n_iters
	
    auc_score = []
        
    kf = model_selection.StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    for f, (train_idx, val_idx) in tqdm(enumerate(kf.split(x, y))):
        df_train, df_val = x.iloc[train_idx], x.iloc[val_idx]
        train_target, val_target = y[train_idx], y[val_idx]
        dtrain = lgbm.Dataset(df_train, train_target)
        dvalid = lgbm.Dataset(df_val, val_target, reference=dtrain)
        
        model = lgbm.train(
                lgb_params, 
                dtrain, 
                num_boost_round, 
                valid_sets=dvalid, 
                verbose_eval=False
	)
           
        predicted = model.predict(df_val)
        auc  = roc_auc_score(val_target, predicted)
        auc_score.append(auc)
      
    return np.mean(auc_score)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--path", type=str, default="../")
    parser.add_argument("--filename", type=str, default="train.csv")
    parser.add_argument("--n_trials", type=float, default=100, required=False)
    parser.add_argument("--es_stop", type=int, default=100, required=False)
    parser.add_argument("--n_iters", type=int, default=200, required=False)
    
    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    df = pd.read_csv(os.path.join(args.path, args.filename))
    optimize_func=partial(optimize,x=df, y=df['target'].values)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_func, n_trials=args.n_trials)
    trial = study.best_trial
    print('Score: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))
