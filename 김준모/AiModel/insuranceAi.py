import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# dataset import
data = pd.read_csv('insurance.csv')

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

# input feature 와 output feature 분리
data_x = data.drop(['charges'], axis=1)
data_y = data['charges']

# train set 과 test set 을 분리한다. 8:2의 비율로 무작위로 추출하였다.
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2, shuffle=True, random_state=7)

# test 의 output 값을 csv 파일로 추출한다.
y_test.to_csv('insurance_testOutput.csv', index=False)

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

# 학습된 모델을 이용하여 test set 에 대하여 predict 를 진행한다.
pred = grid_GBR.predict(X_test)

# predict 한 결과값을 csv 파일로 저장한다.
pd.DataFrame(pred).to_csv('insurance_predict.csv', index=False)

# predict 한 값과 실제 output 값을 이용하여 RMSE 와 MAE 를 구한다.
RMSE = mean_squared_error(y_test, pred) ** 0.5
MAE = mean_absolute_error(y_test, pred)

# RMSE와 MAE를 출력한다.
print('the RMSE of the predict is ', RMSE)
print('the MAE of the predict is ', MAE)
