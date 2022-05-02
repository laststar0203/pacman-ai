import numpy as np
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

# 훈련용 데이터를 생성한 후 케라스의 입력에 맞게 차원을 수정(샘플 개수, 입력 개수)
train_x = np.linspace(0, np.pi * 2, 20).reshape(20, 1)

# 훈련용 정답 데이터
train_Y = np.sin(train_x)

# 테스트용 데이터를 생성한 후 케라스의 입력에 맞게 차원을 수정(샘플 개수, 입력 개수)
test_x = np.linspace(0, np.pi * 2, 60).reshape(60,1)

# 테스트용 정답 데이터
test_y = np.sin(test_x)

plt.plot(test_y)

print(test_y)

exit()

# 입력층의 뉴런 수
input_node = 1

# 중간층의 뉴런 수
hidden_node = 3

# 출력층의 뉴런 수
output_node = 1

# 모델 초기화
model = nn.Sequential(
    nn.Linear(1, 3),
    nn.Sigmoid(),
    nn.Linear(3,1)
)

alpha = 0.1

# 4가지(sgd, momentum, RMSprop, Adam) 최적화 방법을 딕셔너리로 정의
optimizer_option = {
    'sgd': optim.SGD(lr=0.1),
    'momentum': optim.SGD(lr=alpha, momentum=0.9),
    'RM-Sprop': optim.RMSprop(lr=0.01),
    'Adam': optim.Adam(lr=0.01)
}


# 최적화 방법별 학습 진행 과정을 저장
result = []

# 최적화 방법별 훈련 데이터 결과 저장
train_y_hat = []

# 최적화 방법별 테스트 데이터 결과를 저장
test_y_hat = []

criterion = nn.MSELoss()

for optimizer_name, optimizer in optimizer_option.items():

    for epoch in range(50000):
        output = model(train_x)
        loss = criterion(output, train_Y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}\t Loss: {loss.item()}')

    pred = model(test_x)

