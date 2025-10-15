# 중간고사 - regression

## 머신러닝

- 학습(learning)의 정의 : 기계가 프로그램 코드에 의존하지 않고 스스로 목표를 찾아 나아가도록 하는 행위
- 목표 : 오차(error)를 최소화 할 수 있는 파라미터(θ)를 결정하여 가설함수(h(θ))를 정의
- 가설함수(h(θ)) : 학습된 모델의 최종 산출물, 새로운 입력에 대해 예측 또는 분류를 수행

### 머신러닝의 알고리즘

- **지도학습**
    - label이 있는 훈련 데이터를 사용한다.
    - 명확한 답을 가지고 학습하므로 가장 좋은 성능을 가진다.
    - ex) regression(회귀), classification(분류)
- **비지도학습**
    - label이 없어, 데이터 간의 거리를 기반으로 유사한 데이터끼리 clustering(비슷한 특성을 가진 데이터들의 집단)한다.
    - 데이터 그룹화(Grouping) 및 특징 추출
- **강화학습**
    - 주어진 환경에서 행동을 수행하고 관찰하여 그 결과 얻게되는 보상을 최대화 하는 방향으로 행동을 결정한다.

## 회귀모델

- **Linear Regression(선형회귀)**

$$
min_\theta J(\theta)
$$

- 목적함수(J(θ))를 최소화하는 θ를 결정한다.

$$
J(\theta) = \frac{1}{2} \sum_{i=1}^{m} (h_{\theta}(x^{i}) - y^{i})^2
$$

- 학습 방법 : object function → cost funtion ⇒ loss funtion(딥러닝)

- **Probabilistic Interpretation(확률적 해석)**
    - 회귀모델에서 오차(ϵ)가 가우시안 분포(Normal Distribution)을 따른다
    - Normal distribution은 이상적인 형태
    
    $$
    \begin{align*}& y^{i} = \theta^T x^{i} + \epsilon^{i} \\& \epsilon^{i} \sim N(0, \sigma^2) \\& p(\epsilon^{i}) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(\epsilon^{i})^2}{2\sigma^2}\right) \\& p(y^{i}|x^{i};\theta) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{i} - \theta^T x^{i})^2}{2\sigma^2}\right) \\& \text{즉, } y^{i}|x^{i};\theta \sim N(\theta^T x^{i}, \sigma^2)\end{align*}
    $$
    
    - likelihood함수 = 우도함수 = L(θ) ⇒ 모든 데이터( input data X)를 곱한다.
    
    $$
    \begin{align*}L(\theta) &= p(y|X;\theta) \\&= \prod_{i=1}^{m} p(y^{i}|x^{(i)};\theta) \\&= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{i} - \theta^T x^{i})^2}{2\sigma^2}\right)\end{align*}
    $$
    
    - likelihood 함수를 단순화하기 위해 log를 취한다. ⇒ log likelihood
    - log를 취해도 경향에 대한 정보는 바뀌지 않는다.

$$
\begin{align*}l(\theta) &= \log L(\theta) \\&= \log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{i} - \theta^T x^{i})^2}{2\sigma^2}\right) \\&= \sum_{i=1}^{m} \log \left[ \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{i} - \theta^T x^{i})^2}{2\sigma^2}\right) \right] \\&= m\log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2} \sum_{i=1}^{m} (y^{i} - \theta^T x^{i})^2\end{align*}
$$

Maximum likelihood estimation :  L(θ)를 최대화 하기 위한 θ를 선택하는 것이다.

Log likelihood 함수를 최대화 하기 위해서는 $\frac{1}{2} \sum_{i=1}^{m} (y^{i} - \theta^Tx^{i})^2$ 이 항을 최소화 해야한다.

- **Gradient descent(GD) = 경사하강법**
    - θ에 대해 J(θ)를 편미분 하여 기울기를 구한다.
    - 반복적(Iterative)으로 기울기의 반대 방향으로 θ를 업데이트 하여 최소지점을 구한다.
    - α를 통해 θ변화량을 조정한다.
    
    $$
    \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
    $$
    
    $$
    \theta_j := \theta_j + \alpha \sum_{i=1}^{m} (y^{i} - h_{\theta}(x^{i}))x_{j}^{i}
    $$
    
- **Batch Gradient descent (BGD)**
- 모든 훈련 데이터 (M개)를 사용하여 θ를 한번에 업데이트 한다.
- 가장 정확한 최적화된 θ값에 도달할 수 있다.
- 연산량과 비용이 많이 들어 시간이 오래걸린다.

$$
\begin{align*}& \text{Repeat until convergence \{} \\& \qquad \theta_j := \theta_j + \alpha \sum_{i=1}^{m} (y^{i} - h_{\theta}(x^{i}))x_{j}^{i} \quad \text{(for every j)} \\& \text{\}}\end{align*}
$$

- `∑` (시그마) 기호 존재 : 모든 훈련 데이터(m개)의 오차를 전부 계산하고 평균을 내어 파라미터를 단 한 번 업데이트하는 방식

- **Stochastic Gradient descent (SGD) : 확률적 경사 하강법**
    - 연산량과 비용이 많이 드는 문제를 해결하기 위한 알고리즘
    - 전체 데이터를 사용하는 것이 아닌 랜덤하게 추출된 일부 데이터 샘플을 사용하여 θ를 업데이트한다. 이 과정을 여러번 반복하여 최적의 θ값을 찾는다.
    - 연산랸을 크게 줄이고, 최적의 θ값에 대한 근사점에 빠르게 도달할 수 있다.
    - 최소값에 완전히 도달하지 못하고 근사값을 구할 수 있다.
    
    $$
    \begin{align*}
    & \text{Loop \{} \\
    & \qquad \text{for } i=1 \text{ to } m \text{ \{} \\
    & \qquad \qquad \theta_j := \theta_j + \alpha (y^{i} - h_{\theta}(x^{i}))x_{j}^{i} \quad \text{(for every j)} \\
    & \qquad \text{\}} \\
    & \text{\}}
    \end{align*}
    $$
    
    - `∑` 기호가 없다. 하나의 훈련 데이터 샘플(i번째)에 대한 오차만 계산하여 파라미터를 즉시 업데이트하는 방식
    - SGD는 실제 AI 어플(딥러닝)에서 전체 데이터 셋 대신 Mini-Batch 형태로 샘플링하여 학습량을 획기적으로 줄일 수 있다.

- **Normal Equation(정규 방정식)**
    - J(θ)를 θ에 대해 미분하여 미분한 값이 0이 되는 지점을 행렬 연산하여 한번에 계산해 최소 지점으로 이동하는 것
    - 장점 : 반복하는것 없이 가장 정확한 최적의 θ값을 한번에 찾는다.
    - 행렬의 역행렬((X^TX)^-1) 계산에 많은 연산량이 필요하여 데이터 셋이 많아질 경우 연산량이 기하급수적으로 증가하여 비효율적이다.

$$
\begin{align*}
& J(\theta) = \frac{1}{2}\sum_{i=1}^{m}(h_{\theta}(x^i) - y^i)^2 \\
& \nabla_{\theta}J(\theta) \approx \vec{0} \\
\\
& X = \begin{bmatrix} (x^1)^T \\ (x^2)^T \\ \vdots \\ (x^m)^T \end{bmatrix},
& X\theta = \begin{bmatrix} (x^1)^T \theta \\ (x^2)^T \theta \\ \vdots \\ (x^m)^T \theta \end{bmatrix} = \begin{bmatrix} h_{\theta}(x^1) \\ h_{\theta}(x^2) \\ \vdots \\ h_{\theta}(x^m) \end{bmatrix}
\end{align*}
$$

$$
\begin{align*}& J(\theta) = \frac{1}{2}(X\theta - y)^T (X\theta - y) \\\\& X\theta - y = \begin{bmatrix} h_{\theta}(x^1) - y^1 \\ h_{\theta}(x^2) - y^2 \\ \vdots \\ h_{\theta}(x^m) - y^m \end{bmatrix}\end{align*}
$$

$$
\begin{align*}\nabla_{\theta}J(\theta) &= \nabla_{\theta}\frac{1}{2}(X\theta - y)^T (X\theta - y) \\&= \frac{1}{2}\nabla_{\theta}(\theta^T X^T - y^T)(X\theta - y) \\&= \frac{1}{2}\nabla_{\theta}[\theta^T X^T X\theta - \theta^T X^T y - y^T X\theta + y^T y] \\&= \frac{1}{2}[2X^T X\theta - X^T y - X^T y] \\&= X^T X\theta - X^T y \approx 0\end{align*}
$$

$$
\begin{align*}& X^T X\theta = X^T y \quad \text{(normal equation)} \\\\& \therefore \theta = (X^T X)^{-1} X^T y\end{align*}
$$

→ 행결계산식 풀이

## 학습 모델의 적합성

- **Underfitting(과소 적합)**
    - 모델이 데이터의 특징을 제대로 반영하지 못해 학습이 덜 된 상태
- **Overfitting(과대 적합)**
    - 모델이 훈련 데이터에 너무 과하게 일치되어, 새로운 데이터에 대한 예측 성능이 떨어지는 상태
- **Locally Weighted Regression(LWR)**
    - 데이터 분포가 비선형이거나 Overfiting 문제 있을 때, 전체 데이터가 아닌 특정 지역에만 선형 회귀를 적용하여 정확도를 높인다.
    - new input X를 중심으로 W(가중치 함수)를 적용해 가까운 샘플에만 높은 가중치를 부여하고 먼 샘플은 무시하여 지역적으로 최적의 θ를 찾는다.
    
    $$
    \begin{align*}& \text{Fit } \theta \text{ to minimize } \sum_{i} w^{(i)}(y^{(i)} - \theta^T x^{(i)})^2 \\& \text{Output } \theta^T x\end{align*}
    $$