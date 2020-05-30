# Testing out various regression models and comparing their scores against each other using cross-validation

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, SGDRegressor
models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    KNeighborsRegressor(),
    SVR(),
    SGDRegressor()
]
for model in models:
    scores= cross_val_score(model, X, y, scoring = "neg_mean_squared_error", cv = 5)
    real_scores = np.sqrt(-scores)
    print(f"The scores for {model.__class__.__name__} were {real_scores} and the average was {np.average(real_scores)}")
    print("-------------------------------------------------")
    
"""The scores for LinearRegression were [3.64734249 4.0007618  3.55820396 3.13155557 3.44229933] and the average was 3.5560326309596753
-------------------------------------------------
The scores for DecisionTreeRegressor were [4.52130256 4.85685506 4.09978282 4.70135509 5.11109457] and the average was 4.6580780216230995
-------------------------------------------------
The scores for RandomForestRegressor were [3.36865968 3.94736274 2.99496026 2.94677825 3.36925849] and the average was 3.325403883616553
-------------------------------------------------
The scores for KNeighborsRegressor were [3.90109003 4.21803044 3.56409031 3.15577314 3.3776461 ] and the average was 3.64332600300265
-------------------------------------------------
The scores for SVR were [3.69551184 4.01723239 3.62499587 2.97981152 3.16413683] and the average was 3.496337688284099
-------------------------------------------------
The scores for SGDRegressor were [3.64275516 3.98085489 3.49706336 3.10637595 3.38017201] and the average was 3.521444275502467
-------------------------------------------------"""
    
# The model that gave the best score was the RandomForestRegressor so we shall go ahead with that
model = RandomForestRegressor()
model.fit(X, y)

# Do some hyperparameter tuning to try to improve the neg_mean_squared_error using Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [{'n_estimators': [100, 150, 200], 'min_samples_split': [2, 3, 4], 'min_samples_leaf': [1, 2, 3], 
               'ccp_alpha':  [0.1, 0.01, 0.001, 0.0], 'min_weight_fraction_leaf': [0.00, 0.01, 0.02, 0.03], 'random_state': [0, 42, 69]}]
grid_search = GridSearchCV(estimator = model, 
                           param_grid = parameters, 
                           scoring = 'neg_mean_squared_error',
                           cv = 5)
grid_search = grid_search.fit(X, y)
print(grid_search.best_score_)
print(grid_search.best_params_)

"""
-10.890504222006847
{'ccp_alpha': 0.0, 'min_samples_leaf': 2, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 150, 'random_state': 0}"""

print(np.sqrt(-grid_search.best_score_))
# 3.30007639638946

# Thus, we can fit the model again with the new parameters. Co-incidentally, most of the parameters are with their default values
# So we will only need to change the n_estimators and random_state

model = RandomForestRegressor(n_estimators = 150, random_state = 0)
model.fit(X, y)

test_X = full_pipeline.transform(test_dataset)
predictions = model.predict(test_X)

# Upon submitting it on Kaggle, I got an RMSE of 2.72 which is a good value
# You can try using an ANN and some more feature selection/extraction to get a better score
