# -*- coding: utf-8 -*-
# !/usr/bin/python
# Ref : https://jimmy-ai.tistory.com/32
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
# 데이터셋 로드
iris = load_iris()
df = pd.DataFrame(data= np.c_[iris.data, iris.target] ,
                  columns= ['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

from sklearn.model_selection import train_test_split
# train, test 데이터셋 분리
X = df[df.columns[:-1]]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
# 정규화 작업
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
# SVM 모델 생성
model = SVC(kernel='poly', C = 3, degree = 3)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
# test 데이터셋도 정규화(train 데이터셋 기준으로 학습시킨 정규화 모듈 사용)
X_test = scaler.transform(X_test)
y_pred = model.predict(X_test) # 예측 라벨
accuracy_score(y_test, y_pred)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# SVM 결과로 시각화(PCA 2차원 축소 후 결과 확인)
pca = PCA(n_components=2)

# test 데이터셋 기준 시각화 진행
X_test_pca = pca.fit_transform(X_test)
y_find = y_test.reset_index(drop = True)

# target 마다 index 가져오기(꽃 종류마다 색깔을 다르게 시각화 목적) : 실제 라벨 기준
index_0 = y_find[y_find == 0].index
index_1 = y_find[y_find == 1].index
index_2 = y_find[y_find == 2].index

# target 마다 index 가져오기(꽃 종류마다 색깔을 다르게 시각화 목적) : 예측 라벨 기준
y_pred_Series = pd.Series(y_pred)
index_0_p = y_pred_Series[y_pred_Series == 0].index
index_1_p = y_pred_Series[y_pred_Series == 1].index
index_2_p = y_pred_Series[y_pred_Series == 2].index

# 시각화
plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.scatter(X_test_pca[index_0, 0], X_test_pca[index_0, 1], color = 'purple', alpha = 0.6, label = 'setosa')
plt.scatter(X_test_pca[index_1, 0], X_test_pca[index_1, 1], color = 'green', alpha = 0.6, label = 'versicolor')
plt.scatter(X_test_pca[index_2, 0], X_test_pca[index_2, 1], color = 'yellow', alpha = 0.6, label = 'virginica')
plt.title('Real target', size = 13)
plt.legend()

plt.subplot(122)
plt.scatter(X_test_pca[index_0_p, 0], X_test_pca[index_0_p, 1], color = 'purple', alpha = 0.6, label = 'setosa')
plt.scatter(X_test_pca[index_1_p, 0], X_test_pca[index_1_p, 1], color = 'green', alpha = 0.6, label = 'versicolor')
plt.scatter(X_test_pca[index_2_p, 0], X_test_pca[index_2_p, 1], color = 'yellow', alpha = 0.6, label = 'virginica')
plt.title('SVM result', size = 13)
plt.legend()
plt.show()