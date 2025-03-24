import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 다항 회귀의 입력값을 변환하기 위한 모듈을 불러옵니다.
from sklearn.preprocessing import PolynomialFeatures

def load_data():
    
    np.random.seed(0)
    
    X = 3*np.random.rand(50, 1) + 1
    y = X**2 + X + 2 +5*np.random.rand(50,1)
    
    return X, y
    
"""
1. PolynomialFeature 객체를 활용하여 
   각 변수 값을 제곱하고, 
   데이터에 추가하는 함수를 구현합니다.
"""

def Polynomial_transform(X):
    
    poly_feat = None
    
    poly_X = None
    
    print("변환 이후 X 데이터\n",poly_X[:3])
    
    return poly_X
    
"""
2. 다중 선형회귀 모델을 불러오고, 
   불러온 모델을 학습용 데이터에 맞추어 
   학습시킨 후 해당 모델을 반환하는 
   함수를 구현합니다.
"""

def Multi_Regression(poly_x, y):
    
    multilinear = None
    
    None
    
    return multilinear
    
    
# 그래프를 시각화하는 함수입니다.
def plotting_graph(x,y,predicted):
    fig = plt.figure()
    plt.scatter(x, y)
    
    plt.scatter(x, predicted,c='r')
    plt.savefig("test.png")
    
    
"""
3. 모델 학습 및 예측 결과 확인을 위한 
   main 함수를 완성합니다.
   
   학습이 완료된 모델을 활용하여 
   테스트 데이터에 대한 예측을 수행합니다.
"""
def main():
    
    X,y = load_data()
    
    poly_x = Polynomial_transform(X)
    
    linear_model = Multi_Regression(poly_x,y)
    
    predicted = None
    
    plotting_graph(X,y,predicted)
    
    return predicted
    
if __name__=="__main__":
    main()
