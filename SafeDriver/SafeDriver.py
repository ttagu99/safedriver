import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import gc
import lightgbm as lgb

root = r'D:\kaggle'
train=pd.read_csv(root+'/input/train/train.csv', na_values=-1)
test=pd.read_csv(root+'/input/test/test.csv', na_values=-1)

features = train.drop(['id','target'], axis=1).values
targets = train.target.values

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897
def gini(actual, pred, cmpcol = 0, sortcol = 1):
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)
 
def gini_normalized(a, p):
    return gini(a, p) / gini(a, a)

def gini_xgb(preds, dtrain):
    labels = dtrain.get_label()
    gini_score = gini_normalized(labels, preds)
    return 'gini', gini_score

def gini_(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y)) ], dtype=np.float)
    g = g[np.lexsort((g[:,2], -1*g[:,1]))]
    gs = g[:,0].cumsum().sum() / g[:,0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)


def gini_lgb(preds, dtrain):
    y = list(dtrain.get_label())
    score = gini_(y, preds) / gini_(y, y)
    return 'gini', score, True

unwanted = train.columns[train.columns.str.startswith('ps_calc_')]
train = train.drop(unwanted, axis=1)  
test = test.drop(unwanted, axis=1)  

kfold = 5
skf = StratifiedKFold(n_splits=kfold, random_state=42)


# More parameters has to be tuned. Good luck :)
params = {
    'min_child_weight': 10.0,
    'objective': 'binary:logistic',
    'max_depth': 7,
    'max_delta_step': 1.8,
    'colsample_bytree': 0.4,
    'subsample': 0.8,
    'eta': 0.025,
    'gamma': 0.65,
    'num_boost_round' : 700
    }

X = train.drop(['id', 'target'], axis=1).values
y = train.target.values
test_id = test.id.values
test = test.drop('id', axis=1)



sub = pd.DataFrame()
sub['id'] = test_id
sub['target'] = np.zeros_like(test_id)

# lgb
params = {'metric': 'auc', 'learning_rate' : 0.01, 'max_depth':10, 'max_bin':10,  'objective': 'binary', 
          'feature_fraction': 0.8,'bagging_fraction':0.9,'bagging_freq':10,  'min_data': 500}
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('[Fold %d/%d]' % (i + 1, kfold))
    X_train, X_eval = X[train_index], X[test_index]
    y_train, y_eval = y[train_index], y[test_index]
    lgb_model = lgb.train(params, lgb.Dataset(X_train, label=y_train), 2000, 
                  lgb.Dataset(X_eval, label=y_eval), verbose_eval=100, 
                  feval=gini_lgb, early_stopping_rounds=100)
    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
    sub['target'] += lgb_model.predict(test.values, num_iteration=lgb_model.best_iteration) / (kfold)
gc.collect()

#for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#    print('[Fold %d/%d]' % (i + 1, kfold))
#    X_train, X_valid = X[train_index], X[test_index]
#    y_train, y_valid = y[train_index], y[test_index]
#    # Convert our data into XGBoost format
#    d_train = xgb.DMatrix(X_train, y_train)
#    d_valid = xgb.DMatrix(X_valid, y_valid)
#    d_test = xgb.DMatrix(test.values)
#    watchlist = [(d_train, 'train'), (d_valid, 'valid')]

#    # Train the model! We pass in a max of 1,600 rounds (with early stopping after 70)
#    # and the custom metric (maximize=True tells xgb that higher metric is better)
#    mdl = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)

#    print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))
#    # Predict on our test data
#    p_test = mdl.predict(d_test, ntree_limit=mdl.best_ntree_limit)
#    sub['target'] += p_test/(2*kfold)

#gc.collect()
sub.head()
sub.to_csv(root + '/LGBStratifiedKFold_63_10.csv', index=False)
