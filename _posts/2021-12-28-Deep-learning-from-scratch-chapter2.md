---
date: 2021-12-28 00:00:00 +0900
title:  "Deep Learning from Scratch Chapter2 정리"
subtitle: "자연어와 단어의 분산 표현"
categories: Deep-learning
tags: Machine-learning Deep-learning NLP
comments: true
use_math: true

---

> 상세한 파이썬 코드는 생략하였음.

# Chapter2 자연어와 단어의 분산 표현

## 2.1 자연어 처리란

**자연어(natural language)** 란 한국어와 영어 등 우리가 평소에 쓰는 말이다.

**자연어 처리(Natural Language Processing)** 란 사람의 말(자연어)을 컴퓨터에게 이해시키기 위한 기술이다.

사람의 말을 컴퓨터가 이해하도록 만들어서, 컴퓨터가 사람에게 도움이 되는 일을 수행하게 하는 것이 목적이다.

### 2.1.1 단어의 의미

사람의 말은 **문자** 로 구성되며, 말의 의미는 **단어** 로 구성된다. 따라서 자연어를 컴퓨터에게 이해시키는 데는 **단어의 의미** 를 이해시키는 것이 중요하다.

Chapter2의 주제는 **단어의 의미 이해시키기** 이며 아래 두 가지 기법을 살펴본다.

- 시소러스(유의어 사전)를 활용한 기법

- 통계 기반 기법

## 2.2 시소러스

**시소러스** 란 유의어 사전으로, `뜻이 같은 단어(동의어)` 나 `뜻이 비슷한 단어(유의어)` 가 한 그룹으로 분류되어 있다.

![](/assets/5a1e20651f62fb3716d1839e5813ac1c26e48188.png)

또한 자연어 처리에 이용되는 시소러스에서는 단어 사이의 `상위와 하위` 혹은 `전체와 부분` 등, 더 세세한 관계까지 정의해둔 경우가 있다. [그럼 2-2]와 같이 각 단어의 관계를 그래프 구조로 정의한다.

![](/assets/1a6ffbb0c02a81ee4ab78542f55b17a006667ce2.png)

이처럼 모든 단어에 대한 유의어 집합을 만든 다음, 단어들의 관계를 그래프로 표현하여 단어 사이의 연결을 정의할 수 있으며 이 `단어 네트워크` 를 이용하여 컴퓨터에게 단어 사이의 관계를 학습시킬 수 있다.

### 2.2.1 WordNet

자연어 처리 분야에서 가장 유명한 시소러스는 **WordNet** 이며 WordNet을 사용하면 유의어를 얻거나 `단어 네트워크` 를 얻을 수 있고 단어 사이의 유사도를 구할 수도 있다.

### 2.2.2 시소러스의 문제점

- 시대 변화에 대응하기 어렵다.
  
  + 사람이 사용하는 말은 때때로 새로운 단어가 생겨나고, 옛말은 사라지고, 시대에 따라 언어의 의미가 변하기도 한다. 이런 변화에 대응하려면 시소러스를 사람이 수작업으로 끊임없이 갱신해주어야 한다.

- 사람을 쓰는 비용이 크다.
  
  - 영어를 예로 들면, 현존하는 영단어의 수는 1,000만 개가 넘고 이 방대한 단어들 모두에 단어 사이의 관계를 정의하는데에는 엄청난 인적 비용이 발생한다.

- 단어의 미묘한 차이를 표현할 수 없다.
  
  - 시소러스는 뜻이 비슷한 단어들을 묶는데 실제 비슷한 단어들이라도 미묘한 차이가 있다. 시소러스는 이 미묘한 차이를 표현할 수 없다.

위 문제들을 피하기 위해, **통계 기반 기법** 과 신경망을 사용한 **추론 기반 기법(Chapter 3)** 을 주로 다룬다. 이 두 기법에서는 대량의 텍스트 데이터로부터 `단어의 의미` 를 자동으로 추출한다.

## 2.3 통계 기반 기법

통계 기반 기법에선 **말뭉치** 를 사용한다. **말뭉치** 는 자연어 처리 연구나 애플리케이션을 염두에 두고 수집된 대량의 텍스트 데이터이다. 말뭉치에는 `문장을 쓰는 방법, 단어를 선택하는 방법, 단어의 의미` 등 자연어에 대한 `사람의 지식`이 충분히 담겨있다고 볼 수 있다. 통계 기반 기법의 목표는 말뭉치에서 자동으로, 그리고 효율적으로 그 핵심을 파악하는 것이다.

### 2.3.1 말뭉치 전처리하기

파이썬을 이용하여 말뭉치를 전처리한다. 이때, 전처리는 텍스트 데이터를 단어로 분할하고 그 분할된 단어들을 단어 ID 목록으로 변환하는 과정을 말한다. 다음 예시 문장을 활용한다. 이하 상세한 코드는 생략한다.

```python
text = 'You say goodbye and I say hello.'
```

먼저, 예시 문장을 단어 단위로 분할한다.

```python
# 1. 문장 소문자화
# 2. 마침표 앞 공백 삽입
# 3. 문장 분할
# >>> ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
```

- 문장 첫머리의 대문자로 시작하는 단어도 소문자 단어와 똑같이 취급하기 위해 lower() 메서드를 사용하여 모든 문자를 소문자로 변환한다.

- 마침표도 하나의 단어로 활용하기 위해 마침표 앞에 공백을 추가한다.

- 공백을 기준으로 문장을 분할한다.

이로서 예시 문장을 단어 목록 형태로 이용할 수 있다. 다만, 단어를 텍스트 형태 그대로 조작하기엔 여러모로 불편하므로 각 단어에 ID를 부여하고 ID의 리스트로 이용할 수 있도록 전처리한다.

```python
# word_to_id = {단어 : ID}
# id_to_word = {ID : 단어}
# >>> {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
# >>> {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
```

이것으로 단어 ID와 단어의 대응 딕셔너리가 만들어졌다. 이 딕셔너리를 사용하여 `단어 목록` 을 `단어 ID 목록` 으로 변환한다.

```python
# 생성한 단어 목록을 단어 ID 목록으로 변
# >>> array([0, 1, 2, 3, 4, 1, 5, 6])
```

위 모든 과정을 하나의 함수로 통합한다.

```python
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word

text = 'You say goodbye and I say hello.'
coupus, word_to_id, id_to_word = preprocess(text)
```

### 2.3.2 단어의 분산 표현

자연어 처리에선 `단어의 의미` 를 정확하게 파악할 수 있는 벡터 표현이 필요하다. 이를 자연어 처리 분야에선 단어의 **분산 표현** 이라 부른다. 단어의 분산 표현은 단어를 고정 길이의 밀집벡터로 표현한다. 밀집벡터는 원소가 0이 아닌 실수인 벡터를 말한다. 이러한 단어의 분산 표현을 어떻게 구현할 것인가가 이번 장의 중요한 주제이다.

### 2.3.3 분포 가설

단어를 벡터로 표현하는 중요한 기법은 모두 `단어의 의미는 주변 단어에 의해 형성된다.` 라는 아이디어에 뿌리를 두고 있으며 이를 **분포 가설(Distributional Hypothesis)** 이라 한다. 단어 자체에는 의미가 없고 그 단어가 사용된 `맥락` 이 의미를 형성한다는 것이다. `맥락` 이란, 어떤 단어의 주변에 놓인 단어를 의미한다.

![](/assets/c335a4eeb29d7f81ebeaf25ede4a8ee1ab35b043.png)

맥락의 크기를 `윈도우 크기` 라 한다. 상황에 따라 왼쪽 단어 또는 오른쪽 단어만 사용하기도 하며, 문장의 시작과 끝을 고려하기도 한다.

### 2.3.4 동시 발생 항렬

어떤 단어를 기준으로, 그 주변에 어떤 단어가 몇 번이나 등장했는지를 세어 집계하는 방법을 통해 단어를 벡터로 나타내보자. 위 예시 문장과 preprocess() 함수를 활용한다.

```python
text = 'You say goodbye and I say hello.'
coupus, word_to_id, id_to_word = preprocess(text)

print(coupus)
print(id_to_word)
# >>> [0 1 2 3 4 1 5 6]
# >>> {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
```

단어 ID가 0인 `you` 단어의 맥락에 해당하는 단어의 빈도를 세어본다. 이 때 윈도우 크기는 1이다.

![](/assets/332c3d8706f4c2c662ddfddb553ac81057fc4518.png)

윈도우 크기가 1일 때, 맥락은 `say` 단어 한 개 뿐이다.

![](/assets/0a99de28b19fc0644c06dfb2d689a14ed4569511.png)

이를 바탕으로 `you` 라는 단어를 `[0, 1, 0, 0, 0, 0, 0]` 이라는 벡터로 표현할 수 있다. 이같은 방식으로 모든 단어에 대한 맥락의 빈도를 세어 표로 정리한다.

![](/assets/f578bbda93d37e4c880f5f7e476d91b6e04fa354.png)

이 표의 각 행은 해당 단어를 표현한 벡터가 된다. 이 표가 행렬의 형태를 띈다고 해서 **동시 발생 행렬** 이다. 파이썬으로 말뭉치로부터 동시 발생 행렬을 만들어주는 함수를 구현한다.

```python
create_co_matrix(corpus, 7)
"""
array([[0, 1, 0, 0, 0, 0, 0],
       [1, 0, 1, 0, 1, 1, 0],
       [0, 1, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 1, 0]])
"""
```

### 2.3.5 벡터 간 유사도

벡터 간 유사도를 측정하는 다양한 방법이 있다.

- 내적

- 유클리드 거리

- 코사인 유사도

- ETC...

단어 벡터의 유사도를 나타낼 때에는 **코사인 유사도(Cosine Similarity)** 를 자주 이용한다. 코사인 유사도는 다음 식으로 정의된다.

$$similarity(\vec{x}, \vec{y})=\frac{\vec{x}\cdot \vec{y}}{\vert\vert \vec{x}\vert\vert\vert\vert \vec{y}\vert\vert}=\frac{x_1 y_1+\cdots x_n y_n}{\sqrt{x_1^2+\cdots+x_n^2}\sqrt{y_1^2+\cdots+y_n^2}}$$

코사인 유사도의 분자에는 `벡터의 내적` , 분모에는 `각 벡터의 크기` 가 들어간다. 이 때 각 벡터의 크기는 L2 Norm으로 계산한다. 코사인 유사도를 직관적으로 해석하면 `두 벡터가 가리키는 방향이 얼마나 비슷한가` 이다. 두 벡터의 방향이 같다면 코사인 유사도는 1, 반대라면 -1이 된다. 다음은 코사인 유사도를 계산하는 함수이다.

```python
def cos_similarity(x, y, eps=1e-8):
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)

    return np.dot(nx, ny)
```

두 벡터 중 한 벡터가 제로 벡터가 들어오면 `ZeroDivisionError` 가 발생하기 때문에 아주 작은 값인 `Epsilon` 을 더해준다. 다음은 `you` 와 `i` 의 코사인 유사도를 계산하는 코드이다.

```python
text = 'You say goodbye and I say hello.'
coupus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']] # 'you'의 단어 벡터
c1 = C[word_to_id['i']] # 'i'의 단어 벡터
print(cos_similarity(c0, c1))
# >>> 0.7071067811865475
```

### 2.3.6 유사 단어의 랭킹 표시

어떤 단어가 검색어로 주어지면, 그 검색어와 비슷한 단어를 유사도 순으로 출력하는 함수를 구현한다.

![](/assets/d49bff4c08c7fcd7bac23f6f9042d14b244c51d1.png)

most_similar() 함수는 다음과 같은 과정으로 동작한다.

1. 검색어의 단어 벡터를 꺼낸다.
2. 검색어의 단어 벡터와 다른 모든 단어 벡터의 코사인 유사도를 각각 계산한다.
3. 계산한 코사인 유사도 결과를 기준으로 값이 높은 순서대로 출력한다.

`you` 를 검색어로 지정해 유사한 단어들을 추출하면 다음과 같은 결과를 출력한다.

```python
most_similar('you', word_to_id, id_to_word, C, top=5)
"""
[query] you
 goodbye: 0.7071067758832467
 i: 0.7071067758832467
 hello: 0.7071067758832467
 say: 0.0
 and: 0.0
"""
```

## 2.4 통계 기반 기법 개선하기

### 2.4.1 상호정보량

동시 발생 횟수를 척도로 사용하는 것은 고빈도 단어(the, a ) 때문에 그리 좋은 척도가 되지 못한다. 이 문제를 해결하기 위해 **점별 상호정보량(Pointwise Mutual Information, PMI)** 이라는 척도를 사용한다. PMI는 확률 변수 x, y에 대해 다음과 같이 정의한다.

$$
PMI(x, y)=log_{2}{P(x,y)\over P(x)P(y)}
$$

동시 발생 행렬을 사용하여 위 PMI 식을 재정의한다. C는 동시 발생 행렬, N은 말뭉치에 포함된 단어의 수이다.

$$
PMI(x, y)=log_{2}{P(x,y)\over P(x)P(y)}=log_{2}{\frac{C(x, y)}{N}\over \frac{C(x), C(y)}{N}}=log_{2}{C(x,y)N\cdot\over C(x)C(y)}
$$