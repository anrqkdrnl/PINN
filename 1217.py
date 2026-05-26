import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

# CUDA 컨텍스트 강제 생성 (더미 연산 추가)
torch.cuda.init()
torch.randn(1, device="cuda")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # 🔹 인덱스 값으로 설정

# 물리적 상수
rho = 1000.0  # 밀도 kg/m^3
cp = 1000.0   # 비열 J/(kg·K)
k = 10.0      # 열전도율 W/(m·K)

# 경계 및 초기 조건
T_left = 50.0     
T_right = 150.0   
T_initial = 50.0  

L = 0.1           
t_sim = 200.0     

# 신경망 클래스 정의
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            nn.init.xavier_uniform_(self.layers[-1].weight)
        self.activation = torch.tanh

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# PDE 손실 함수
def pde_loss(model, x_pde):
    x_pde = x_pde.detach().clone().requires_grad_(True)  # leaf로 만들기
    T_pred = model(x_pde)

    T_x_t = autograd.grad(T_pred, x_pde, torch.ones_like(T_pred), create_graph=True)[0]
    T_x = T_x_t[:, 0:1]  
    T_t = T_x_t[:, 1:2]  
    T_xx = autograd.grad(T_x, x_pde, torch.ones_like(T_x), create_graph=True)[0][:, 0:1]  

    f = (rho * cp * T_t - k * T_xx) / (rho * cp)  
    return torch.mean(f**2)

# 데이터 생성 함수 수정
def generate_data(n_pde, n_bc, n_ic):
    # PDE 점 (랜덤 재생성)
    x_pde = torch.rand(n_pde, 2, device=device, requires_grad=True)  
    
    x_pde = x_pde.clone()  # 🔹 기존 텐서를 복사하여 사용
    x_pde[:, 0] = x_pde[:, 0] * L
    x_pde[:, 1] = x_pde[:, 1] * t_sim

    # 경계 조건 점
    t_bc = torch.linspace(0, t_sim, n_bc, device=device).unsqueeze(1)
    x_bc_left = torch.cat([torch.zeros_like(t_bc), t_bc], dim=1)
    x_bc_right = torch.cat([L * torch.ones_like(t_bc), t_bc], dim=1)
    T_bc_left = T_left * torch.ones(n_bc, 1, device=device)
    T_bc_right = T_right * torch.ones(n_bc, 1, device=device)

    # 초기 조건 점
    x_ic = torch.cat([torch.linspace(0, L, n_ic, device=device).unsqueeze(1), torch.zeros(n_ic, 1, device=device)], dim=1)
    T_ic = T_initial * torch.ones(n_ic, 1, device=device)

    return x_pde, x_bc_left, x_bc_right, T_bc_left, T_bc_right, x_ic, T_ic


# 모델 생성
layers = [2, 256, 256, 256, 256, 1]  
model = PINN(layers).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3000)

epochs = 50000
losses = []

for epoch in range(epochs):
    x_pde, x_bc_left, x_bc_right, T_bc_left, T_bc_right, x_ic, T_ic = generate_data(4000, 100, 100)

    optimizer.zero_grad()
    loss_pde = pde_loss(model, x_pde)
    loss_bc_left = torch.mean((model(x_bc_left) - T_bc_left)**2)
    loss_bc_right = torch.mean((model(x_bc_right) - T_bc_right)**2)
    loss_ic = torch.mean((model(x_ic) - T_ic)**2)

    loss = loss_pde + loss_bc_left + loss_bc_right + loss_ic
    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    losses.append(loss.item())
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item()}")

# 손실 그래프 출력
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.yscale('log')
plt.show()

x = torch.linspace(0, L, 100, device=device)
t = torch.linspace(0, t_sim, 100, device=device)
X, T = torch.meshgrid(x, t, indexing='ij')

x_test = torch.hstack((X.flatten()[:, None], T.flatten()[:, None])).to(device)
with torch.no_grad():
    T_pred = model(x_test).cpu().numpy()
T_pred = T_pred.reshape(100, 100)

plt.figure(figsize=(8, 6))
plt.contourf(X.cpu(), T.cpu(), T_pred, levels=100, cmap='hot')
plt.colorbar(label='Temperature (°C)')
plt.xlabel('Position x (m)')
plt.ylabel('Time t (s)')
plt.title('Temperature Distribution')
plt.savefig('temperature_distribution_1217_1702.png')
plt.show()

# 모델 및 결과 저장
torch.save(model.state_dict(), 'pinn_model_optimized_1217_1702.pth')
np.save('loss_values_optimized.npy', np.array(losses))
print("Model, loss curve, and predictions saved successfully!")
