---
date: 2021-10-20 00:00:00 +0900
title:  "Swallow neural networks 가중치와 편향을 반복문 없이 업데이트하기"
subtitle: "Swallow neural networks 가중치와 편향을 반복문 없이 업데이트하기"
categories: Machine-learning
tags: Machine-learning Neural-network Deep-learning
comments: true
use_math: true
---
# Swallow neural networks 가중치와 편향을 반복문 없이 업데이트하기

## 1. train 함수 수정
train 함수 내에서 가중치와 편향을 반복문을 통해 업데이트하는 코드를 반복문 없이 동작하도록 수정한다.

```python
def train(X, Y, model, lr=0.1):
    dW1 = np.zeros_like(model.W1)
    db1 = np.zeros_like(model.b1)
    dW2 = np.zeros_like(model.W2)
    db2 = np.zeros_like(model.b2)
    m = len(X)

    cost = 0.0

    for x, y in zip(X, Y):
        a2, (z1, a1, z2, _) = model.predict(x)
        if y == 1:
            cost -= np.log(a2)
        else:
            cost -= np.log(1-a2)

        diff = a2-y

        db2 += diff

        #for i in range(model.num_hiddens):
        #    dW2[i] += a1[i]*diff
        dW2 += a1*diff

        #for i in range(model.num_hiddens):
        #    db1[i] += (1-a1[i]**2)*model.W2[i]*diff
        db1 += (1-a1**2)*model.W2*diff

        #for i in range(model.num_hiddens):
        #    for j in range(model.num_input_features):
        #        dW1[i,j] += x[j]*(1-a1[i]**2)*model.W2[i]*diff
        dW1 += np.outer(x, (1-a1**2)*model.W2*diff).T

    cost /= m
    model.W1 -= lr * dW1/m
    model.b1 -= lr * db1/m
    model.W2 -= lr * dW2/m
    model.b2 -= lr * db2/m

    return cost
```
- `dW2` 행렬를 업데이트하는 첫번째 반복문에서 `a1` 행렬에 diff 스칼라 값을 곱하면 `a1` 행렬의 각 원소에 `diff` 가 곱해지고 `dW2` 행렬과 `a1` 행렬은 서로 shape 가 같기 때문에 반복문없이 두 행렬을 더하여도 각 위치에 맞는 원소끼리 더해진다.
- `db1` 행렬을 업데이트하는 두번째 반복문에서도 위와 마찬가지로 동작한다.
- `dW1` 행렬을 업데이트하는 세번째 반복문은 `dW1` 행렬이 2차원 행렬이기 때문에 이중 반복문을 이용하여 업데이트 하였다. 이는 `np.outer` 함수(행렬곱)를 이용하여 계산하고 `transpose` 하는 것으로 `dW1` 행렬 계산을 대체할 수 있다.

## 2. 결과
```python
for epoch in range(100):
    cost = train(X, Y, model, 1.0)
    if epoch%10 == 0:
        print(epoch, cost)

#0 [0.70577315]
#10 [0.68634607]
#20 [0.67255153]
#30 [0.63927949]
#40 [0.58035322]
#50 [0.50978504]
#60 [0.44864856]
#70 [0.40426057]
#80 [0.37439698]
#90 [0.35469361]
```
```python
print(model.predict((1,1))[0].item())
print(model.predict((1,0))[0].item())
print(model.predict((0,1))[0].item())
print(model.predict((0,0))[0].item())

#0.09654599868346721
#0.8305469780858583
#0.8535266138864402
#0.09703222351725922
```
