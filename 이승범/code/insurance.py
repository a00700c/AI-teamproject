import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("insurance.csv")

# 데이터 전처리 과정
# 성별이 남자면 0, 여자면 1 로 설정한다.
data.loc[data.sex == 'male', 'sex'] = '0'
data.loc[data.sex == 'female', 'sex'] = '1'
# 비흡연자는 0, 흡연자는 1 로 설정한다.
data.loc[data.smoker == 'no', 'smoker'] = '0'
data.loc[data.smoker == 'yes', 'smoker'] = '1'
# 지역에 따라 0, 1, 2, 3 으로 나눈다.
data.loc[data.region == 'southwest', 'region'] = '1'
data.loc[data.region == 'southeast', 'region'] = '2'
data.loc[data.region == 'northwest', 'region'] = '3'
data.loc[data.region == 'northeast', 'region'] = '4'


data_x = data.drop(['charges'], axis=1)
data_y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=7)

y_test.to_csv('insurance_testOutput.csv', index=False)

params = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8,],  # 트리의 최대 깊이
    'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8],  # 리프노드가 되기 위헤 필요한 최소한의 샘플 데이터수
    'min_samples_split': [3, 4, 6, 7, 8],  # 노드를 분할하기 위한 초소한의 샘플 데이터수
    'n_estimators': [52, 53, 54, 55, 56, 57, 58, 59, 60]  # 결정트리의 개수
}

RFR = RandomForestRegressor()

grid_tree = GridSearchCV(RFR, param_grid=params, cv=2, n_jobs=-1)

grid_tree.fit(X_train, y_train)

print('best parameters : ', grid_tree.best_params_)

em = grid_tree.best_estimator_
pred = em.predict(X_test)

pd.DataFrame(pred).to_csv('insurance_predict.csv', index=False)

RMSE = np.sqrt(mean_squared_error(y_test, pred))
MAPE = np.mean(np.abs((y_test - pred) / y_test)) * 100

print('the RMSE of the predict is ', RMSE)
print('the MAPE of the predict is ', MAPE)
