# Use XGBoost to predict the continuous data
# XGBoost is an incredibly strong ML model that has started making waves in the ML field recently.
# More on XGBoost can be found here: https://xgboost.readthedocs.io/en/latest/index.html

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X, y)

test_X = full_pipeline.transform(test_dataset)
predictions = regressor.predict(test_X)

# With the XGBoost model, I got an RMSE of 2.66 which is a definite improvement on my RandomForestRegressor model.
# However, I feel this score can be improved further by feature selection/extraction.
