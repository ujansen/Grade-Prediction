# I would recommend looking at the individual xgboost and lgbm .py files before looking at this

import lightgbm as lgbm
lgbm_train = lgbm.Dataset(X, y)

parameters = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'learning_rate': 0.03,
    'feature_fraction': 0.9,
    'bagging fraction': 0.8,
    'bagging freq': 5
}

lgbm_model = lgbm.train(parameters, lgbm_train, num_boost_round = 130)

from xgboost import XGBRegressor
xgb_model = XGBRegressor()
xgb_model.fit(X, y)

test_X = full_pipeline.transform(test_dataset)
grades = lgbm_model.predict(test_X, num_iteration = lgbm_model.best_iteration)
grades += xgb_model.predict(test_X)
grades /= 2

# With this ensemble model, I got an RMSE of 2.57.
# This could possibly be further improved by feature selection/extraction or even using an ANN
