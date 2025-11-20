#-*- coding: utf-8 -*-
#!/usr/bin/python
# Ref : https://planharry.tistory.com/43

# 필요한 패키지 설치
import pandas as pd
import numpy as np
# iris 데이터 불러오기 위한 datasets 설치
from sklearn import datasets

# skearn.datasets에 포함된 iris(붓꽃) 데이터 가져오기
iris = datasets.load_iris()
# iris 데이터 내 data값들
data = pd.DataFrame(iris.data) ; data
# iris데이터의 feature 이름
feature = pd.DataFrame(iris.feature_names) ; feature
# data의 컬럼명을 feature이름으로 수정하기
data.columns = feature[0]
# 세가지 붓꽃의 종류
target = pd.DataFrame(iris.target) ; target
# 컬럼명 바꾸기
target.columns=['target']
# data와 target 데이터프레임을 합치기 (axis=1, columns으로 합치기)
df = pd.concat([data, target], axis=1)
df.head()
df.info()

#target 컬럼을 object 타입으로 변경
df = df.astype({'target': 'object'})
# 결측치 없음, 각 속성마다 150개 row씩 있음
df.describe()
# 클러스터 돌리기 전 변수를 생성
df_f = df.copy()

import seaborn as sns
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
sns.pairplot(df_f, hue="target")
plt.show()

# 2차원 그리기
fig = plt.figure(figsize=(5, 5))
X = df_f
plt.plot(X.iloc[:, 0]
         , X.iloc[:, 3]
         , 'o'
         , markersize=2
         , color='green'
         , alpha=0.5
         , label='class1')

plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend() # 범례표시
plt.show()

# 3차원 그리기
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X = df_f

# 3d scatterplot 그리기
ax.scatter(X.iloc[:, 0]
           , X.iloc[:, 1]
           , X.iloc[:, 2]
           , c=X.index #마커컬러
           , s=10 #사이즈
           , cmap="Oranges" #컬러맵
           , alpha=1 #투명도
           , label='class1' #범례
           )
plt.legend() #범례표시
plt.show()

from sklearn.cluster import KMeans
# 적절한 군집수 찾기
# Inertia(군집 내 거리제곱합의 합) value (적정 군집수)
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(df_f)
    inertias.append(model.inertia_)
# Plot ks vs inertias
plt.figure(figsize=(4, 4))
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# K-Means 모델과 군집 예측값을 생성
# 클러스터 모델 생성 파라미터는 원할 경우 추가
clust_model = KMeans(n_clusters = 3 # 클러스터 갯수
                     , n_init=10 # initial centroid를 몇번 샘플링한건지, 데이터가 많으면 많이 돌릴수록안정화된 결과가 나옴
                     , max_iter=500 # KMeans를 몇번 반복 수행할건지, K가 큰 경우 1000정도로 높여준다
                     , random_state = 42
                     , algorithm='auto'
                     )
# 생성한 모델로 데이터를 학습시킴
clust_model.fit(df_f) # unsupervised learning

# 결과 값을 변수에 저장
centers = clust_model.cluster_centers_ # 각 군집의 중심점
pred = clust_model.predict(df_f) # 각 예측군집
print(pd.DataFrame(centers))
print(pred[:10])

# 원래 데이터에 예측된 군집 붙이기
clust_df = df_f.copy()
clust_df['clust'] = pred
clust_df.head()

# scaling하지 않은 데이터를 학습하고 시각화하기
plt.figure(figsize=(20, 6))
X = clust_df
plt.subplot(131)
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], data=df_f, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,1], c='black', alpha=0.8, s=150)
plt.subplot(132)
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,2], data=df_f, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,2], c='black', alpha=0.8, s=150)
plt.subplot(133)
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,3], data=df_f, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers[:,0], centers[:,3], c='black', alpha=0.8, s=150)
plt.show()

# 3차원으로 시각화하기
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
X = clust_df
# 데이터 scatterplot
ax.scatter(X.iloc[:,0]
           , X.iloc[:,1]
           , X.iloc[:,2]
           , c = X.clust
           , s = 10
           , cmap = "rainbow"
           , alpha = 1
           )
# centroid scatterplot
ax.scatter(centers[:,0],centers[:,1],centers[:,2] ,c='black', s=200, marker='*')
plt.show()

cluster_mean = clust_df.groupby('clust').mean()
cluster_mean

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
scaled_df = pd.DataFrame(standard_scaler.fit_transform(df_f.iloc[:,0:4]), columns=df_f.iloc[:,0:4].columns) # scaled된 데이터

# create model and prediction
# clust_model은 스케일링 전 fit과 동일하게 맞춤
clust_model.fit(scaled_df) # unsupervised learning #애초에 결과를 모르기 때문에 data만 넣어주면 됨

centers_s = clust_model.cluster_centers_
pred_s = clust_model.predict(scaled_df)

# 스케일링 전에 합쳐준 데이터프레임에 스케일한 군집 컬럼 추가하기
clust_df['clust_s'] = pred_s
clust_df

# scaling 완료한 데이터를 학습하고 시각화하기
plt.figure(figsize=(20, 6))
X = scaled_df
plt.subplot(131)
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,1], data=scaled_df, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers_s[:,0], centers_s[:,1], c='black', alpha=0.8, s=150)
plt.subplot(132)
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,2], data=scaled_df, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers_s[:,0], centers_s[:,2], c='black', alpha=0.8, s=150)
plt.subplot(133)
sns.scatterplot(x=X.iloc[:,0], y=X.iloc[:,3], data=scaled_df, hue=clust_model.labels_, palette='coolwarm')
plt.scatter(centers_s[:,0], centers_s[:,3], c='black', alpha=0.8, s=150)
plt.show()

# 스케일링 전 데이터의 군집
pd.crosstab(clust_df['target'], clust_df['clust'])
# 스케일링 후 데이터의 군집
pd.crosstab(clust_df['target'], clust_df['clust_s'])