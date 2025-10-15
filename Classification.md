# 중간고사 - Classification

## 분류 모델(Classification)

목표 : 데이터가 어떤 그룹(0 or 1)에 속하는지를 분류(Classification)하는 것이다.

⇒ 이진 분류 모델, 연속적인 값을 예측하는 선형 회귀와 다름

선형회귀가 분류 모델에 부적함

why? 연속적인 값을 예측하므로 분류 문제에 적용하면, 새로운 데이터가 추가될 때마다 회귀선(분류 기준선)이 크게 바뀔 수 있다.

- **Logistic Regression**
    - 선형함수 ($\theta^Tx$)의 결과값을 바로 출력하는 것이 아닌 sigmoid function(g(z))에 입력하여 출력값을 0과 1 사이의 확률 값으로 변환
        
        $$
        h_{\theta}(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
        $$
        
    - $h_\theta(x)$≥0.5일때 Y=1 / $h_\theta(x)$<0.5일때, Y=0
    - z→∞일 때, g(z) = 1 / z→-∞일 때, g(z) = 0
        
        $$
        \begin{align*}& p(y=1|x;\theta) = h_{\theta}(x), \quad p(y=0|x;\theta) = 1 - h_{\theta}(x) \\& p(y|x;\theta) = h_{\theta}(x)^y (1 - h_{\theta}(x))^{1-y}, \quad y \in \{0,1\} \\& y=1 \text{ 이면 } p(y|x;\theta) = h_{\theta}(x) \text{, } y=0 \text{ 이면 } p(y|x;\theta) = 1 - h_{\theta}(x)\end{align*}
        $$
        
    
    $$
    p(y=1|x) = h_{\theta}(x) \text{ , } p(y=0|x) = 1 - h_{\theta}(x)
    $$
    
- **logistic Regression 최적화**
    - logistic regression model은 least squares regression처럼 maximum likelihood를 통해 유도 될 수 있다.
        - 오차를 최소화 하고 Likelihood를 최대화(Maximize)하는 θ값을 찾는다.
            
            $$
            L(\theta)= \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(y^{i} - \theta^T x^{i})^2}{2\sigma^2}\right)
            $$
            
            → y의 값에 따라 1 or 0이 이항분포의 모습을 띄는 형태이다
            
    - why? classification에서는 예측값(y hat)이 중요한 것이 아닌 input X가 어떤 그룹(0 or 1)에 속하는지 판단하는것이 중요하다
    - L(θ)를 단순화 하기 위해 log를 취해준다.
        
        $$
        \begin{align*}l(\theta) &= \log L(\theta) \\&= \sum_{i=1}^{m} \left[ y^{i}\log h_{\theta}(x^{i}) + (1 - y^{i})\log(1 - h_{\theta}(x^{i})) \right]\end{align*}
        $$
        
    - log likelihood 함수(l(θ)를 최대화 하기 위해 Gradient ascent를 사용한다
    - Gradient ascent vs Gradient descent : 개념 자체는 다르지 않으며, ascent는 최대값 descent는 최소값을 찾는것 / 두 함수는 대칭이며 손실 함수의 형태를 조정하면 동일한 매커니즘으로 사용이 가능
        
        $$
        \theta_j := \theta_j + \alpha \frac{\partial}{\partial \theta_j} l(\theta)
        $$
        
- **회귀모델과 분류 모델의 공통점**
    - 내가 목표하는 object를 최소화, 최대화를 하여 내가 하고자 하는 행위에 대한 $\theta$를 결정한 후 $\theta$를 가지고 새로운 데이터인 input X에 대한 값을 결정한다.
    - 기존의 가설함수 $h_\theta(x)$를 이용하여 예측된 y값을 구하는 것이다.
- **회귀 모델과 분류 모델의 차이점**
    - 회귀 : y에 대한 확실한 값이 필요하고 에러를 최소화 하는 패턴 함수를 사용한다.
    - 분류 : 그룹단위로 나위었을 때 데이터의 패턴이 어떤 경향을 보이는지, 언제 최대의 경향을 보여줄 수 있는지에 관한 패턴함수를 만들어 사용한다. 또한 예측된 y값이 중요한 것이 아닌 input X가 계산되어지는 결과를 통해 T/F 판별이 중요하다.

- **Newton’s Method**
    - θ를 업데이트 하는 또 다른 최적화 방법
    - 2차 미분(H) 정보를 활용하여 최적점에 더 빨리 도달한다.
    
    $$
    \theta := \theta - H^{-1} \nabla_{\theta} l(\theta)
    $$
    
    - Greadient Descent보다 반복 횟수가 적어(Iterative하지 않고 직관적으로) 매우 빠르게 수렴할 수 있다.
    - Hessian 행렬의 계산(2차 비분)에 많은 연산 비용이 발생해 일반적으로 Gradient descent가 더 많이 사용된