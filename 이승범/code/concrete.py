import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Concrete_Data_Yeh.csv")

data_x = data.drop(['csMPa'], axis=1)
data_y = data['csMPa']

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=7)

y_test.to_csv('concrete_testOutput.csv', index=False)

params = {
    'max_depth': [76, 77, 78, 79, 80, 81, 82, 83, 84],  # 트리의 최대 깊이
    'min_samples_leaf': [1, 2, 3, 4],  # 리프노드가 되기 위헤 필요한 최소한의 샘플 데이터수
    'min_samples_split': [2, 3, 4, 5],  # 노드를 분할하기 위한 초소한의 샘플 데이터수
    'n_estimators': [84, 85, 86, 87, 88, 89, 90, 91, 92]  # 결정트리의 개수
}

RFR = RandomForestRegressor()

grid_tree = GridSearchCV(RFR, param_grid=params, cv=2, n_jobs=-1)

grid_tree.fit(X_train, y_train)

print('best parameters : ', grid_tree.best_params_)

em = grid_tree.best_estimator_
pred = em.predict(X_test)

pd.DataFrame(pred).to_csv('concrete_predict', index=False)

RMSE = np.sqrt(mean_squared_error(y_test, pred))
MAPE = np.mean(np.abs((y_test - pred) / y_test)) * 100

print('the RMSE of the predict is ', RMSE)
print('the MAPE of the predict is ', MAPE)
