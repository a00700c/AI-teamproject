import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# dataset import
data = pd.read_csv('Concrete_Data_Yeh.csv')

# input feature 와 output feature 분리
data_x = data.drop(['csMPa'], axis=1)
data_y = data['csMPa']

# train set 과 test set 을 분리한다. 8:2의 비율로 무작위로 추출하였다.
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=7)

# test 의 output 값을 csv 파일로 추출한다.
y_test.to_csv('concrete_testOutput.csv', index=False)

# GradientBoostingRegressor 을 가져온다
GBR = GradientBoostingRegressor()

# GridSearch 를 사용하기 위해 parameters 를 정한다.
parameters = {'learning_rate': [0.01, 0.02, 0.03, 0.04],
              'subsample': [0.9, 0.5, 0.2, 0.1],
              'n_estimators': [100, 500, 1000, 1500],
              'max_depth': [4, 6, 8, 10]
              }

# GBR 을 모델로, parameters 를 파라미터로 받고, cv는 교차 검증을 위해 분할하는 데이터 셋의 개수이다.
grid_GBR = GridSearchCV(estimator=GBR, param_grid=parameters, cv=2, n_jobs=-1)

# train set 을 학습시킨다.
grid_GBR.fit(X_train, y_train)

# GridsearchCV 를 통해 찾아낸 best parameters 를 출력한다.
print('best parameters : ', grid_GBR.best_params_)

# 학습된 모델을 이용하여 test set 에 대하여 predict 를 진행한다.
pred = grid_GBR.predict(X_test)

# predict 한 결과값을 csv 파일로 저장한다.
pd.DataFrame(pred).to_csv('Concrete_predict.csv', index=False)

# predict 한 값과 실제 output 값을 이용하여 RMSE 와 MAE 를 구한다.
RMSE = mean_squared_error(y_test, pred) ** 0.5

# MAPE 를 구하는 함수이다.
def find_mape(y_test, pred):
    return np.mean(np.abs((y_test - pred) / y_test)) * 100


MAPE = find_mape(y_test, pred)

# RMSE와 MAPE를 출력한다.
print('the RMSE of the predict is ', RMSE)
print('the MAPE of the predict is ', MAPE)