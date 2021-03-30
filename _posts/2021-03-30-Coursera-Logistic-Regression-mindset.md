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

### 1.2 - 데이터셋 준비

#### 1.2.1 - 개요

`("data.h5")` 데이터셋이 주어진다. 이 데이터셋은 다음을 포함하고 있다.

+ `train` : 고양이(y=1) 또는 고양이가 아님(y=0) 라벨을 가지고 있는 이미지 트레이닝셋이다.
+ `test` : 고양이 또는 고양이가 아님 라벨을 가지고 있는 이미지 테스트셋이다.
+ 각 이미지의 사이즈는 `(num_px, num_px, 3)` 이며 3은 이미지의 채널 수(RGB)를 의미한다.
+ 각 이미지는 `height = num_px`, `width = num_px` 이다. 따라서, 각 이미지는 정사각형이다.

#### 1.2.2 - 데이터 불러오기

이제 `lr_utils`에서 임포트한 함수를 통해 데이터셋을 불러온다.

```python
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
```

`load_dataset()` 함수를 이용하여 각 변수에 데이터셋을 저장한다.

+ `train_set_x_orig` : 신경망을 트레이닝할 이미지 데이터셋이다.
+ `train_set_y` : 신경망을 트레이닝할 이미지 데이터셋의 라벨 값들이다.
+ `test_set_x_orig` : 신경망을 테스트할 이미지 데이터셋이다.
+ `test_set_y` : 신경망을 테스트할 이미지 데이터셋의 라벨 값들이다.
+ `classes` : 라벨 값들의 이름(cat / non-cat)이 있는 array이다.

위 변수들은 모두 `numpy.ndarray` 이며,  `ndarray` 는 동일 타입의 원소가 담긴 다차원 행렬로 벡터 연산이 가능하다.

`train_set_x_orig` 과 `test_set_x_orig` 뒤에 `_orig` 가 붙어있는 이유는 이미지 데이터셋을 가공한 후 원본 데이터셋과 구분을 하기 위해 붙어있다.

#### 1.2.3 - 이미지 확인하기

다음 코드를 통해 데이터셋의 이미지 파일 하나를 출력할 수 있다.

```python
# Example of a picture
index = 2
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
```
![Logistic_Regression_image1](/assets/Logistic_Regression_image1.png)

