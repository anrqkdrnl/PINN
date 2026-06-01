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
    torch.cuda.set_device(0)  # 인덱스 값으로 설정

# 물리적 상수
rho = 1000.0  # 밀도 kg/m^3
cp = 1000.0   # 비열 J/(kg·K)
k = 10.0      # 열전도율 W/(m·K)
alpha = k / (rho * cp) # 열확산계수(Thermal diffusivity)

# 경계 및 초기 조건
T_left = 50.0     
T_right = 150.0   
T_initial = 50.0  

L = 0.1           
t_sim = 200.0     

# 🔹 1. 이론해(Analytical Solution) 계산 함수
def exact_solution(x, t, num_terms=100):
    # 정상상태(Steady-state) 온도 분포: 50 + (150-50)/0.1 * x
    T_ss = 50.0 + 1000.0 * x  
    T_transient = np.zeros_like(x)
    
    # 과도상태(Transient) 푸리에 급수 해
    for n in range(1, num_terms + 1):
        Bn = (200.0 / (n * np.pi)) * ((-1)**n)
        T_transient += Bn * np.sin(n * np.pi * x / L) * np.exp(-alpha * (n * np.pi / L)**2 * t)
        
    return T_ss + T_transient

# 🔹 신경망 클래스 정의 (정규화 추가)
class PINN(nn.Module):
    def __init__(self, layers, lb, ub):
        super(PINN, self).__init__()
        
        # 최솟값(lb)과 최댓값(ub)을 텐서로 변환하여 저장
        self.lb = torch.tensor(lb, dtype=torch.float32, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=device)
        
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            nn.init.xavier_uniform_(self.layers[-1].weight)
        self.activation = torch.tanh

    def forward(self, x):
        # 🔹 입력 데이터 정규화 (Min-Max Scaling -> [-1, 1])
        x = 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0
        
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x

# PDE 손실 함수
def pde_loss(model, x_pde):
    x_pde = x_pde.detach().clone().requires_grad_(True)
    T_pred = model(x_pde)

    T_x_t = autograd.grad(T_pred, x_pde, torch.ones_like(T_pred), create_graph=True)[0]
    T_x = T_x_t[:, 0:1]  
    T_t = T_x_t[:, 1:2]  
    T_xx = autograd.grad(T_x, x_pde, torch.ones_like(T_x), create_graph=True)[0][:, 0:1]  

    f = (rho * cp * T_t - k * T_xx) / (rho * cp)  
    return torch.mean(f**2)

# 데이터 생성 함수
def generate_data(n_pde, n_bc, n_ic):
    x_pde = torch.rand(n_pde, 2, device=device, requires_grad=True)  
    x_pde = x_pde.clone()  
    x_pde[:, 0] = x_pde[:, 0] * L
    x_pde[:, 1] = x_pde[:, 1] * t_sim

    t_bc = torch.linspace(0, t_sim, n_bc, device=device).unsqueeze(1)
    x_bc_left = torch.cat([torch.zeros_like(t_bc), t_bc], dim=1)
    x_bc_right = torch.cat([L * torch.ones_like(t_bc), t_bc], dim=1)
    T_bc_left = T_left * torch.ones(n_bc, 1, device=device)
    T_bc_right = T_right * torch.ones(n_bc, 1, device=device)

    x_ic = torch.cat([torch.linspace(0, L, n_ic, device=device).unsqueeze(1), torch.zeros(n_ic, 1, device=device)], dim=1)
    T_ic = T_initial * torch.ones(n_ic, 1, device=device)

    return x_pde, x_bc_left, x_bc_right, T_bc_left, T_bc_right, x_ic, T_ic

# 🔹 2. 모델 생성 및 파일명 이름표 설정
layers = [2, 256, 256, 256, 256, 256, 1]  
arch_str = "_".join(map(str, layers))  

# 🔹 하한값(lb)과 상한값(ub) 설정: [x_min, t_min], [x_max, t_max]
lb = [0.0, 0.0]
ub = [L, t_sim]

# 모델 초기화 시 lb, ub 전달
model = PINN(layers, lb, ub).to(device)

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
    
    # 3. 1000번마다, 그리고 마지막 에포크(49999)에 도달했을 때 출력
    if epoch % 1000 == 0 or epoch == (epochs - 1):
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# ==========================================
# 4. 학습 종료 후 시각화 및 결과 일괄 저장
# ==========================================

# (1) 전체 Loss 그래프 딱 1번 그리고 저장
plt.figure(figsize=(6, 4))
plt.plot(losses, color='blue', linewidth=1.5)
plt.xlabel('Epoch')
plt.ylabel('Total Loss (MSE)')
plt.yscale('log')
plt.title(f'Training Loss Curve (Layers: {arch_str})')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f'loss_curve_{arch_str}.png')  
plt.show()

# (2) 결과 분석을 위한 시공간 격자 데이터 준비
x = torch.linspace(0, L, 100, device=device)
t = torch.linspace(0, t_sim, 100, device=device)
X, T = torch.meshgrid(x, t, indexing='ij')

x_test = torch.hstack((X.flatten()[:, None], T.flatten()[:, None])).to(device)
with torch.no_grad():
    T_pred = model(x_test).cpu().numpy()
T_pred = T_pred.reshape(100, 100)

# CPU 메모리로 내려서 numpy로 변환 및 오차 계산
X_np = X.cpu().numpy()
T_np = T.cpu().numpy()
T_exact = exact_solution(X_np, T_np)
Error = np.abs(T_pred - T_exact)

# (3) 1x3 서브플롯으로 PINN 결과, 이론해, 오차 한눈에 비교
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

c1 = axes[0].contourf(X_np, T_np, T_pred, levels=100, cmap='hot')
fig.colorbar(c1, ax=axes[0])
axes[0].set_title('PINN Prediction')
axes[0].set_xlabel('Position x (m)')
axes[0].set_ylabel('Time t (s)')

c2 = axes[1].contourf(X_np, T_np, T_exact, levels=100, cmap='hot')
fig.colorbar(c2, ax=axes[1])
axes[1].set_title('Analytical Solution')
axes[1].set_xlabel('Position x (m)')
axes[1].set_ylabel('Time t (s)')

c3 = axes[2].contourf(X_np, T_np, Error, levels=100, cmap='viridis')
fig.colorbar(c3, ax=axes[2])
axes[2].set_title('Absolute Error')
axes[2].set_xlabel('Position x (m)')
axes[2].set_ylabel('Time t (s)')

plt.tight_layout()
plt.savefig(f'pinn_vs_analytical_{arch_str}.png')  
plt.show()

# (4) 모델 가중치 및 손실값 기록 저장
torch.save(model.state_dict(), f'pinn_model_{arch_str}.pth')
np.save(f'loss_values_{arch_str}.npy', np.array(losses))
print(f"Model and results saved successfully with architecture [{arch_str}]!")