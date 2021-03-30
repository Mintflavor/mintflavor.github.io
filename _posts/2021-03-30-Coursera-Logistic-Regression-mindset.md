---
title: "Coursera Deep Learning Logistic Regression with Neural Network"
date: 2021-03-30 12:28:00 +0900
categories: Deep-Learning
tags: Coursera Deep-Learning Logistic-Regression Neural-Network Keras
---

---
Coursera에서 `Andrew Ng`의 `Neural Networks and Deep Learning`을 수강하고 있다.  
2주차 마지막에 Logistic Regression과 Neural Network를 이용하여 주어진 사진들이 고양이인지 아닌지를 분류하는 신경망을 구현하는 과제가 있다.

> Welcome to the first (required) programming exercise of the deep learning specialization. In this notebook you will build your first image recognition algorithm. You will build a cat classifier that recognizes cats with 70% accuracy!

강좌에서 제공하는 Jupyter Notebook 파일을 차근차근 따라가면 약 70%의 정확성을 가지는 멋진 신경망을 구현할 수 있다.

## 1 - 로지스틱 회귀분석과 신경망을 이용하여 고양이 사진 분류하기
### 1.1 - Package
먼저 데이터 전처리, 로지스틱 회귀분석, 신경망 제작을 위해 필요한 패키지들을 임포트한다.

```python
#Import packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline
```

+ [h5py](http://www.h5py.org) 패키지는 HDF5 바이너리 데이터 포맷을 사용하기 위한 인터페이스 패키지이다.
+ lr_utils 패키지는 신경망 학습을 위한 고양이 사진 데이터셋이 들어있는 패키지이다.

### 1.2 - 데이터셋 개요
`("data.h5")` 데이터셋이 주어진다. 이 데이터셋은 다음을 포함하고 있다.

+ `m_train` : 고양이(y=1) 또는 고양이가 아님(y=0) 라벨을 가지고 있는 이미지 트레이닝셋이다.
+ `m_test` : 고양이 또는 고양이가 아님 라벨을 가지고 있는 이미지 테스트셋이다.
+ 각 이미지의 사이즈는 `(num_px, num_px, 3)` 이며 3은 이미지의 채널 수(RGB)를 의미한다.
+ 각 이미지는 `height = num_px`, `width = num_px` 이다. 따라서, 각 이미지는 정사각형이다.

이제 `lr_utils`에서 임포트한 함수를 통해 데이터셋을 불러온다.

```python
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```
