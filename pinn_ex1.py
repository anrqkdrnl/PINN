import torch
import torch.nn as nn
import torch.optim as optim
import random

# -----------------------------------------------------------------
# Step 1. Generate 10000(=m) train samples (데이터셋 생성), 주어진 값
# -----------------------------------------------------------------
m = 10000
x1_train, x2_train, y_train = [], [], []

for i in range(m):
    x1 = random.uniform(-10, 10) #x1으로만 데이터 분류함 
    x2 = random.uniform(-10, 10) #x2를 만드는 이유 : 모델이 x2는 아무 의미 없는 데이터를 잘 구분해서 가중치를 학습할 수 있는 지를 테스트하기 위해 의도적으로 한 세팅
    
    x1_train.append(x1)
    x2_train.append(x2)
    
    # x1이 -5보다 작거나 5보다 크면 1, 아니면 0
    if x1 < -5 or x1 > 5:
        y_train.append(1.0)
    else:
        y_train.append(0.0)

# PyTorch 연산을 위해 Python 리스트를 Tensor로 변환합니다.
# 모델의 입력 X는 (10000, 2) 형태, 정답 Y는 (10000, 1) 형태로 맞춰줍니다.
X = torch.tensor(list(zip(x1_train, x2_train)), dtype=torch.float32)
Y = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)


# -----------------------------------------------------------------
# Step 2. Initialize W=[w1, w2] and b (모델 및 가중치 초기화)
# -----------------------------------------------------------------
# 로지스틱 회귀는 선형 결합(Linear) 후 시그모이드(Sigmoid)를 통과하는 구조입니다.
# nn.Linear(2, 1)을 선언하면 내부적으로 W와 b가 무작위(small random values)로 자동 초기화됩니다.
model = nn.Sequential( #여러개의 연산을 순서대로 연결해주는 장치 여기서는 nn.Linear -> nn.Sigmoid 순으로 계산하도록 해줌
    nn.Linear(2, 1), #(입력의 개수, 출력의 개수) -> 계산할 행렬의 크기를 정함 여기서 w,b가 무작위로 주어짐
    nn.Sigmoid() #0-1 확률로 변환    
)


# -----------------------------------------------------------------
# Step 3. Determine the learning rate alpha (학습률 설정)
# -----------------------------------------------------------------
learning_rate = 0.01  # 1/100, alpha

# Loss 함수: Binary Cross Entropy Loss
criterion = nn.BCELoss() #nn.BCELoss => L(y,y_hat) = -(ylog(Y_hat)+(1-y)log(1-y_hat)) -> 여기서 그냥 다 더하고 평균까지 내버림 => J

# 최적화 기법(Optimizer): 경사하강법(SGD)을 사용하여 W와 b를 업데이트합니다.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# -----------------------------------------------------------------
# Step 4. Update W, b with 'm' samples for 5000 iterations (학습 루프)
# -----------------------------------------------------------------
iterations = 5000

for i in range(1, iterations + 1):
    # 1. Forward pass: 입력 데이터를 모델에 통과시켜 예측값(확률) 계산
    y_pred = model(X)
    
    # 2. Step 4-2: Cost(Loss) 계산
    loss = criterion(y_pred, Y)
    
    # 3. Backward pass: 경사하강법을 통한 가중치 업데이트
    optimizer.zero_grad()  # 이전 루프에서 계산된 기울기 초기화
    loss.backward()        # Loss를 바탕으로 W와 b에 대한 기울기(Gradient) 계산
    optimizer.step()       # W와 b 업데이트 (학습)
    
    # Step 4-1 & 4-3: 500번 반복할 때마다 상태 출력
    if i % 500 == 0:
        # 현재 학습된 W와 b 값 가져오기
        # model[0]은 첫 번째 층인 nn.Linear(2, 1)을 가리킵니다.
        w1 = model[0].weight[0][0].item()
        w2 = model[0].weight[0][1].item()
        b = model[0].bias[0].item()
        
        # 예측한 확률(y_pred)이 0.5보다 크면 1, 아니면 0으로 분류
        predicted_classes = (y_pred > 0.5).float()
        
        # 실제 정답(Y)과 예측 분류(predicted_classes)가 일치하는 개수 계산
        correct_predictions = (predicted_classes == Y).sum().item()
        
        # 정확도(Accuracy) 계산 (%)
        accuracy = (correct_predictions / m) * 100
        
        # 결과 출력
        print(f"Iteration: {i:4d} | Cost: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
        print(f"  -> W = [{w1:.4f}, {w2:.4f}], b = {b:.4f}")
        print("-" * 65)

        #이 코드는 은닉층이 1개라 1개의 기준을 나눌 수 있음 -> -10 ~ -5~5 ~ 10 은 2개의 기준을 가지고 나누어야 하기 때문에 층을 하나 더해주어야 함