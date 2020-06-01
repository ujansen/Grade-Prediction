# Use LightGBM to predict the continuous data
# XGBoost is an incredibly strong ML model that has started making waves in the ML field recently.
# More on XGBoost can be found here: https://lightgbm.readthedocs.io/en/latest/index.html

# Since we're using LGBM, we need to convert it into a dataset first
import lightgbm as lgbm
lgbm_train = lgbm.Dataset(X, y)

# We need to specify our hyperparameters
parameters = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging fraction': 0.8,
    'bagging freq': 5
}

regressor = lgbm.train(parameters, lgbm_train, num_boost_round = 130)

test_X = full_pipeline.transform(test_dataset)
predictions = regressor.predict(test_X, num_iteration = regressor.best_iteration)
