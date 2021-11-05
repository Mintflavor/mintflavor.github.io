---
date: 2021-10-07 00:00:00 +0900
title:  "K-means vs. DBSCAN"
subtitle: "K-means vs. DBSCAN"
categories: Machine-learning
tags: Machine-learning Clustering
comments: true
use_math: true
---
# K-means 와 DBSCAN 알고리즘 비교
##### 2017010055 박현일

## 1. 데이터셋
K-means 와 DBSCAN 알고리즘 비교를 위해 numpy random 함수를 이용하여 간단한 데이터셋을 만들었다.
![스크린샷 2021-10-07 오전 10.18.01](/assets/스크린샷%202021-10-07%20오전%2010.18.01.png)

## 2. K-means
K-means 알고리즘은 클러스터의 개수를 미리 지정하여야 하기 때문에 `n_cluster` 를 `3` 으로 지정하였다.
```python
K_means = KMeans(n_clusters=3, random_state=0)
K_means.fit(X)
```
![스크린샷 2021-10-07 오전 10.24.56](/assets/스크린샷%202021-10-07%20오전%2010.24.56.png)

각 클러스터의 중심을 기준으로 클러스터링이 잘 이루어진 모습을 확인할 수 있다.

## 3. DBSCAN
6개의 `(eps, min_samples)` 묶음을 준비하여 학습하였고 그 중 가장 성능이 좋은 조합을 발견할 수 있었다.

```python
#enu = ((eps, min_samples))
enu = ((0.5, 3), (0.5, 5), (1, 3), (1, 5), (1.5, 3), (1.5, 5))
```
![스크린샷 2021-10-07 오전 10.48.54](/assets/스크린샷%202021-10-07%20오전%2010.48.54.png)

`eps = 1, min_samples = 3` 일 때 가장 좋은 성능을 보였고 하늘색 클러스터에서 한 개의 이상치(짙은 파란색)를 발견하였다.
