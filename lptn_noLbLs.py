# -*- coding: utf-8 -*-
"""
GRU Encoder + DeepONet Hybrid Model (PI-DeepONet; LPM/T_amb 기반 동적 냉각 버전)
- 입력 시퀀스: [mag, rotor, stator, copper, LPM, T_amb] (정규화 0~1)
- 냉각항: R_coil_amb(상수) 대신 R_ca(t) = 1 / ( h(t) * A_coil ) 적용
  · h(t)는 (우선순위) h_profile → 없으면 LPM, T_amb로부터 경험식으로 생성
  · T_amb도 시간의존(T_amb(t))을 지원 (없으면 상수 self.T_amb 사용)
- 기존 저항 R_pm_coil, R_pm_amb, R_ch_cb 등은 상수 유지

검증 루프에서도 LPM(t), T_amb(t)을 모델에 주입하여 Direct/ Rollout 모두에 반영됨.
"""

# -*- coding: utf-8 -*-
"""
lptn_GRU9_patched.py
- GRU9 기반
- (A) Trunk 입력 확장 [t + controls(t)]
- (B) h 경험식 유지 + 학습 스케일 α 적용
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import matplotlib.pyplot as plt
# ==== Reproducibility Seed (minimal) ====
import os, random
os.environ["PYTHONHASHSEED"] = "43"

try:
    import numpy as np
    np.random.seed(43)
except Exception:
    pass

try:
    import torch
    torch.manual_seed(43)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(43)
except Exception:
    pass
# =======================================


# ---------------------- 설정 ----------------------
DATA_DIR      = "/home/hye/Documents/sung/loss"
DATA_DIR2     = "/home/hye/Documents/sung/gt"
SAVE_DIR_BASE = "/home/hye/Documents/sung"

DEVICE        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PREDICT_TIME  = 3600.0
T_MAX         = 400
TAMB          = 20

# [추가] 시간 가중치 강도 (0이면 기존이랑 동일)
TIME_WEIGHT_ALPHA = 0   # 2~5 사이 값부터 실험 추천


# GRU
GRU_INPUT_DIM  = 7
GRU_HIDDEN_DIM = 64
GRU_N_LAYERS   = 4     # GRU9 기본값 유지
LEARNING_RATE  = 0.001
MAX_ITERS      = 20000
GRAD_CLIP      = 1.0

# ★MOD: 제어 입력 차원
CONTROL_DIM  = 7       # [q_mag,q_rot,q_cop,q_sta,LPM,Tamb,h_dyn]
H_NORM_REF   = 2000.0   # h 정규화 기준값 (W/m²K)




LOSS_MIN_MAX = { #최댓값이 어디서 나온 것인가????
    'mag':  (44.2, 162.7),
    'rot':  (7.0, 107.5),
    'cop':  (55.3, 1811.1),
    'sta':  (139.8, 1070.5),
    'LPM':  (0.0, 1.0),
    'RPM':  (0.0, 8000),
}

torch.set_default_dtype(torch.float64)

# ---------------------- GRU Encoder ----------------------
# [PATCH 1] --- GRUEncoder.forward 수정 ---
# ---------------------- GRU Encoder (DUAL) ----------------------
class GRUEncoder(nn.Module):
    def __init__(self, input_dim=GRU_INPUT_DIM, hidden_dim=GRU_HIDDEN_DIM, n_layers=GRU_N_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # 도메인별 GRU 두 개 (0=variable, 1=fixed)
        self.gru_var = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.gru_fix = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)

        # (권장) 도메인별 learnable 초기 hidden — 콜드스타트 완화
        self.h0_var = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim, dtype=torch.float64))
        self.h0_fix = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim, dtype=torch.float64))

    def _make_h0(self, B: int, device, dtype, is_var: bool):
        base = self.h0_var if is_var else self.h0_fix            # [L,1,H] (float64)
        return base.to(device=device, dtype=dtype).expand(self.n_layers, B, self.hidden_dim).contiguous()

    def forward(self, x, domain_id: torch.Tensor, return_sequences: bool = False):
        """
        x         : [B, N, D]
        domain_id : [B] (0=variable, 1=fixed)
        return_sequences=False -> 마지막 hidden [B,H]
        return_sequences=True  -> 모든 타임스텝 hidden [B,N,H]
        """
        B, N, _ = x.shape
        device, dtype = x.device, x.dtype

        # 출력 버퍼
        y_out = torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype)

        # 배치 내 도메인 마스크
        mask_var = (domain_id == 0)
        mask_fix = (domain_id == 1)

        # variable → gru_var
        if mask_var.any():
            x_var = x[mask_var]                       # [Bv,N,D]
            Bv    = x_var.size(0)
            h0    = self._make_h0(Bv, device, dtype, is_var=True)
            y_var, _ = self.gru_var(x_var, h0)        # [Bv,N,H]
            y_out[mask_var] = y_var

        # fixed → gru_fix
        if mask_fix.any():
            x_fix = x[mask_fix]                       # [Bf,N,D]
            Bf    = x_fix.size(0)
            h0    = self._make_h0(Bf, device, dtype, is_var=False)
            y_fix, _ = self.gru_fix(x_fix, h0)        # [Bf,N,H]
            y_out[mask_fix] = y_fix

        if return_sequences:
            return y_out                               # [B,N,H]
        return y_out[:, -1, :]                         # [B,H]


# ---------------------- Branch Net ----------------------
class BN(nn.Module):
    def __init__(self, input_dim=GRU_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, u):
        return self.net(u)

# ---------------------- ★MOD: Trunk ----------------------
class PM(nn.Module):
    def __init__(self, control_dim=CONTROL_DIM):
        super().__init__()
        in_dim = 1    
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


    def forward(self, t_and_ctrl):
        return self.net(t_and_ctrl)

class COIL(nn.Module):
    def __init__(self, control_dim=CONTROL_DIM):
        super().__init__()
        in_dim = 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)



    def forward(self, t_and_ctrl):
        return self.net(t_and_ctrl)
    
class Rotor(nn.Module):
    def __init__(self, control_dim=CONTROL_DIM):
        super().__init__()
        in_dim = 1     
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


    def forward(self, t_and_ctrl):
        return self.net(t_and_ctrl)
    
class Stator(nn.Module):
    def __init__(self, control_dim=CONTROL_DIM):
        super().__init__()
        in_dim = 1 
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self._init_weights()
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)


    def forward(self, t_and_ctrl):
        return self.net(t_and_ctrl)
    
    
def _to_scalar(x: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    if not torch.is_tensor(x):
        return torch.as_tensor(x, dtype=torch.get_default_dtype())
    if x.ndim == 0:
        return x
    return x.mean() if reduce == "mean" else x.sum()
  

# ---------------------- Hybrid Model ----------------------
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = DEVICE
        self.gru_encoder = GRUEncoder().to(self.device)
        self.BN = BN(input_dim=GRU_HIDDEN_DIM).to(self.device) 
        # 노드별 trunk (t-only)
        self.ROTOR_A = Rotor().to(self.device)
        self.PM      = PM().to(self.device)
        self.ROTOR_B = Rotor().to(self.device)
        self.CS      = COIL().to(self.device)      # slot coil+stator region
        self.STATOR  = Stator().to(self.device)    # 8.06mm pure stator
        self.END     = COIL().to(self.device)      # end winding (t-only)





        self.log_alpha_h   = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=self.device))
        self.log_alpha_rpm = nn.Parameter(torch.tensor(0.0, dtype=torch.float64, device=self.device))

        # self.R_pm_coil = nn.Parameter(
        #     torch.tensor(0.38077977, dtype=torch.float64, device=self.device),
        #     requires_grad=True
        # )

        # 학습 단계 플래그
        # 1: LPTN + Direct를 같이 학습 (지금까지 하던 방식)
        # 2: LPTN은 고정, Direct만 PDE loss 위주로 재튜닝

        # =================================================

        # log_alpha_rpm 도 같이 학습되도록 포함
        # --- optimizer (single, recommended) ---
        # =================================================
        # Optimizers (grouped)
        # =================================================

        # 1) 그룹별 파라미터 수집
        params_seq = []
        params_seq += list(self.gru_encoder.parameters())
        params_seq += list(self.BN.parameters())

        params_pm = list(self.PM.parameters())

        params_rotor = []
        if hasattr(self, "ROTOR_A") and self.ROTOR_A is not None:
            params_rotor += list(self.ROTOR_A.parameters())
        if hasattr(self, "ROTOR_B") and self.ROTOR_B is not None:
            params_rotor += list(self.ROTOR_B.parameters())

        params_cs_end = []
        params_cs_end += list(self.CS.parameters())
        params_cs_end += list(self.END.parameters())

        params_stator = []
        if hasattr(self, "STATOR") and self.STATOR is not None:
            params_stator += list(self.STATOR.parameters())

        params_alpha = [self.log_alpha_h]

        # 2) 중복 제거 유틸 (안전장치)
        def _uniq(params):
            seen = set()
            out = []
            for p in params:
                if p is None:
                    continue
                if not isinstance(p, torch.Tensor):
                    continue
                if not p.requires_grad:
                    continue
                if id(p) in seen:
                    continue
                out.append(p)
                seen.add(id(p))
            return out

        params_seq    = _uniq(params_seq)
        params_pm     = _uniq(params_pm)
        params_rotor  = _uniq(params_rotor)
        params_cs_end = _uniq(params_cs_end)
        params_stator = _uniq(params_stator)
        params_alpha  = _uniq(params_alpha)

        # 3) 그룹별 lr 설정 (필요하면 상단 하이퍼로 빼세요)
        LR_SEQ     = getattr(self, "LR_SEQ", LEARNING_RATE)      # 보통 1e-4
        LR_PM      = getattr(self, "LR_PM", 1e-4)
        LR_ROTOR   = getattr(self, "LR_ROTOR", 1e-4)
        LR_CS_END  = getattr(self, "LR_CS_END", 1e-4)
        LR_STATOR  = getattr(self, "LR_STATOR", 1e-4)
        LR_ALPHA   = getattr(self, "LR_ALPHA", 1e-4)             # alpha는 크게(10~100배)

        # 4) optimizer 생성
        self.opt_seq     = optim.Adam(params_seq,    lr=LR_SEQ)
        self.opt_pm      = optim.Adam(params_pm,     lr=LR_PM)      if len(params_pm)     > 0 else None
        self.opt_rotor   = optim.Adam(params_rotor,  lr=LR_ROTOR)   if len(params_rotor)  > 0 else None
        self.opt_cs_end  = optim.Adam(params_cs_end, lr=LR_CS_END)  if len(params_cs_end) > 0 else None
        self.opt_stator  = optim.Adam(params_stator, lr=LR_STATOR)  if len(params_stator) > 0 else None
        self.opt_alpha   = optim.Adam(params_alpha,  lr=LR_ALPHA)   if len(params_alpha)  > 0 else None

        # 5) 편의: step/zero_grad 호출을 한 번에 하기 위한 리스트
        self.optimizers = [self.opt_seq, self.opt_pm, self.opt_rotor, self.opt_cs_end, self.opt_stator, self.opt_alpha]
        self.optimizers = [o for o in self.optimizers if o is not None]

        # 6) scheduler (선택)
        # - 일단 seq만 스케줄링하는 걸 권장 (alpha까지 코사인으로 줄이면 학습이 죽는 경우가 많음)
        self.scheduler_seq = CosineAnnealingLR(self.opt_seq, T_max=MAX_ITERS, eta_min=1e-7)

        # --- LBFGS: physics(+alpha) fine-tune용 파라미터 묶기 ---
        params_lbfgs = []
        params_lbfgs += params_pm
        params_lbfgs += params_rotor
        params_lbfgs += params_cs_end
        params_lbfgs += params_stator
        params_lbfgs += params_alpha
        params_lbfgs = _uniq(params_lbfgs)

        self.opt_lbfgs_phys = optim.LBFGS(
            params_lbfgs,
            max_iter=4,
            history_size=20,
            line_search_fn="strong_wolfe"
        )


        # (선택) 나중에 필요하면 각 opt에도 scheduler를 따로 달 수 있음
        # self.scheduler_pm = CosineAnnealingLR(self.opt_pm, T_max=MAX_ITERS, eta_min=1e-7) if self.opt_pm else None

       # ============================================================
        # 물리 상수 (논문 geometry + Table 3 재질 기반)  [단위: J/K]
        #   V_coil   = 0.001666 m^3, ρ_cu=8933, cp_cu=386  → C_coil_tot ≈ 5744.6
        #   V_stator = 0.000956 m^3, ρ=7600,  cp=475       → C_stator   ≈ 3451.2
        #   V_rotor+shaft = 0.001204 m^3, ρ=7600, cp=475  → C_rotor    ≈ 4346.4
        #   V_pm     = 0.000156 m^3, ρ=7500,  cp=502       → C_pm       ≈ 587.3
        #   Coil hot : bulk = 0.4 : 0.6 가정
        # ============================================================
        # =========================
        # Thermal capacitances [J/K]  (새 노드 기준)
        # =========================
        C_pm      = 587.34
        C_rotorA  = 2980.39
        C_rotorB  = 658.49
        C_cs      = 4447.28
        C_stator  = 1754.46
        C_end     = 3111.60

        dtype = torch.get_default_dtype()

        # =========================
        # Heat capacities [J/K]  (당신 확정값)
        # =========================
        self.register_buffer('C_pm',     torch.tensor(1000.34,  dtype=dtype))
        self.register_buffer('C_rotorA', torch.tensor(2980.39, dtype=dtype))
        self.register_buffer('C_rotorB', torch.tensor(658.49,  dtype=dtype))
        self.register_buffer('C_CS',     torch.tensor(4447.28, dtype=dtype))
        self.register_buffer('C_stator', torch.tensor(1754.46, dtype=dtype))
        self.register_buffer('C_end',    torch.tensor(3111.60, dtype=dtype))
        self.register_buffer('C_housing', torch.tensor(28823.0319, dtype=dtype))

        # =========================
        # Areas [m^2]  (확정)
        # =========================
        self.register_buffer('A_ra_pm',    torch.tensor(0.0332,   dtype=dtype))
        self.register_buffer('A_pm_rb',    torch.tensor(0.027064, dtype=dtype))
        self.register_buffer('A_rb_cs',    torch.tensor(0.028368, dtype=dtype))
        self.register_buffer('A_cs_st',    torch.tensor(0.039744, dtype=dtype))
        self.register_buffer('A_st_ch',    torch.tensor(0.062832, dtype=dtype))
        self.register_buffer('A_end_surf', torch.tensor(0.014334,  dtype=dtype))  # end-coil 외표면(대류)

        # =========================
        # Conductivities [W/mK]
        # =========================
        self.register_buffer('k_air',   torch.tensor(0.026, dtype=dtype))
        self.register_buffer('k_steel', torch.tensor(42.0,  dtype=dtype))   # 철심(스테이터/로터 철)
        self.register_buffer('k_pm',    torch.tensor(8.9,   dtype=dtype))
        self.register_buffer('k_cu',    torch.tensor(401.0, dtype=dtype))

        # =========================
        # Geometry thickness [m]
        # =========================
        self.register_buffer('L_rotorA', torch.tensor(27.79e-3, dtype=dtype))
        self.register_buffer('L_pm',     torch.tensor(6.07e-3,  dtype=dtype))   # 9.07 → 7.07 반영
        self.register_buffer('L_rotorB', torch.tensor(6.14e-3,  dtype=dtype))
        self.register_buffer('L_CS',     torch.tensor(25.94e-3, dtype=dtype))   # CS 두께(코일+스테이터 겹침 구간)
        self.register_buffer('L_stator', torch.tensor(8.06e-3,  dtype=dtype))

        # Alias(코드에서 L_cs 같은 걸 쓰고 싶으면 반드시 통일)
        self.L_cs = self.L_CS  # optional alias

        # =========================
        # Air-gap params
        # =========================
        self.register_buffer('delta_gap',  torch.tensor(0.6e-3, dtype=dtype))  # 필요시 수정
        self.register_buffer('h_conv0',    torch.tensor(250.0,  dtype=dtype))
        self.register_buffer('h_conv_exp', torch.tensor(0.5,    dtype=dtype))

        # End-coil air convection (상수 시작)
        self.register_buffer('h_air', torch.tensor(1700.0, dtype=dtype))

        # RPM norm
        self.register_buffer("RPM_NORM_REF", torch.as_tensor(8000, dtype=dtype))

        # Housing heat capacity


        # Stator <-> Housing conduction
        R_st_hou = (self.R_st_hou if hasattr(self, "R_st_hou") else torch.tensor(0.030)).to(torch.float64).to(self.device)

        # End <-> Housing conduction (optional but recommended)
        R_end_hou = (self.R_end_hou if hasattr(self, "R_end_hou") else torch.tensor(0.09)).to(torch.float64).to(self.device)

        # Housing -> Ambient convection (must exist for ΔP=0 finite steady state)
        h_out = (self.h_out if hasattr(self, "h_out") else torch.tensor(12.0)).to(torch.float64).to(self.device)
        A_out = (self.A_housing_out if hasattr(self, "A_housing_out") else torch.tensor(0.20)).to(torch.float64).to(self.device)
        R_hou_amb = 1.0 / (h_out * A_out + 1e-12)



        def _R_cond(L, k, A, eps=1e-12):
            # 모두 tensor, scalar 가능
            return (L / (k * A + eps)).to(dtype)
        
        def _h_conv_rot_from_rpm(rpm):
    # rpm: scalar tensor
            rpm_n = torch.clamp(rpm / (self.RPM_NORM_REF + 1e-12), min=0.0)
            return self.h_conv0 * (rpm_n ** self.h_conv_exp)

        def _R_gap_rb_cs(rpm_scalar):
            h_conv_rot = _h_conv_rot_from_rpm(rpm_scalar)
            h_gap_eff  = (self.k_air / (self.delta_gap + 1e-12)) + h_conv_rot
            return 1.0 / (h_gap_eff * self.A_rb_cs + 1e-12)

        # CS 등가 열전도율 (병렬 혼합 가정: k_eff = k_cu*f_cu + k_steel*f_st)
        V_cu_CS = 4.899e-4
        V_st_CS = 7.64e-4
        f_cu = V_cu_CS / (V_cu_CS + V_st_CS)
        f_st = 1.0 - f_cu
        k_CS_eff_val = 401.0 * f_cu + 42.0 * f_st  # ≈ 182 W/mK

        self.register_buffer('k_CS_eff', torch.tensor(k_CS_eff_val, dtype=dtype))


        # RotorA ↔ PM : (로터 철 전도라고 가정)
        self.register_buffer("R_ra_pm", _R_cond(self.L_rotorA, self.k_steel, self.A_ra_pm))

        # PM ↔ RotorB : PM 전도로 가정 (당신 질문대로 “PM 지나가는 부분=42?”는 여기서 제거됨)
        self.register_buffer("R_pm_rb", _R_cond(self.L_pm, self.k_pm, self.A_pm_rb))

        # RotorB ↔ CS : 로터 철 전도(steel)로 가정
        self.register_buffer("R_rb_cs", _R_cond(self.L_rotorB, self.k_steel, self.A_rb_cs))

        # CS ↔ Stator : CS 등가 전도율 사용(임시)
        self.register_buffer("R_cs_st", _R_cond(self.L_CS, self.k_CS_eff, self.A_cs_st))

        # Stator ↔ Channel(냉각채널 벽 전도) : steel 전도로 가정
        self.register_buffer("R_st_ch", _R_cond(self.L_stator, self.k_steel, self.A_st_ch))

        # =========================
        # CS <-> End-coil (axial conduction through copper)
        # =========================
        dtype = torch.get_default_dtype()

        # (당신이 준 부피)
        V_cu_CS = torch.tensor(4.899e-4, dtype=dtype)   # [m^3]
        V_end   = torch.tensor(9.024e-4, dtype=dtype)   # [m^3]

        # 길이: CS는 이미 self.L_CS 사용, End는 새로 정의(필요시 수정)
        self.register_buffer('L_end', torch.tensor(62.93e-3, dtype=dtype))  # [m] 엔드코일 축방향 길이

        # axial 단면적(체적/길이). CS 구리와 End 구리의 단면적이 다를 수 있으니 평균 사용.
        A_ax_cs  = V_cu_CS / (self.L_CS + 1e-12)        # [m^2]
        A_ax_end = V_end   / (self.L_end + 1e-12)       # [m^2]
        A_ax_eff = 0.5 * (A_ax_cs + A_ax_end)

        # 중심-중심 거리(슬랩 모델): 0.5*L_CS + 0.5*L_end
        L_cs_end = 0.5 * self.L_CS + 0.5 * self.L_end

        # R_cs_end = L / (k * A)
        self.register_buffer(
            'R_cs_end',
            (L_cs_end / (self.k_cu * A_ax_eff + 1e-12)).to(dtype)
        )









        # (중요) alpha_sta_to_coil 삭제
        # if hasattr(self, "alpha_sta_to_coil"): delattr(self, "alpha_sta_to_coil")



        # === [추가] PM 대류용 면적 / 스케일 ===
        self.A_pm = torch.tensor(0.00145, dtype=torch.float64)   # PM 유효 면적 (임시값, 나중 조정)
        self.alpha_pm_h = torch.tensor(0.5, dtype=torch.float64)  # 코일 대비 PM 냉각 강도 스케일



        self.register_buffer('A_coil', torch.tensor(0.076))
        self.use_dynamic_h = True
        self.h_profile     = None
        self.LPM_profile   = None
        self.RPM_profile   = None   # ★ 추가 (air-gap에 RPM 필요)
        self.q_profile     = None

        # Tamb는 고정 20°C
        self.T_amb = torch.tensor(20.0, dtype=torch.float64, device=self.device)


    def set_T_amb(self, t_amb):
        # Tamb는 고정이므로 무시
        self.T_amb = torch.tensor(20.0, dtype=torch.float64, device=self.device)


    def _h_gap_eff(self, RPM_t: torch.Tensor):
        """
        RPM_t: [N,1] 또는 [N]
        return: h_gap_eff [N,1]
        """
        dtype = torch.get_default_dtype()
        RPM_t = RPM_t.view(-1, 1).to(self.device).to(dtype)

        # conduction-equivalent through air film
        h_cond = self.k_air / (self.delta_gap + 1e-12)  # [W/m2K]

        # rotation-induced convection term
        rpm_n = RPM_t / (self.RPM_NORM_REF.to(dtype) + 1e-12)
        h_conv_rot = self.h_conv0.to(dtype) * torch.clamp(rpm_n, min=0.0) ** self.h_conv_exp.to(dtype)

        return h_cond + h_conv_rot




    # RPM_ref는 정격 RPM (예: 15000) 정도로 클래스 상단에 하나 정의해두고:
    # self.RPM_REF = 15000.0

    def _h_from_LPM_tcool_rpm_dual(self, LPM_t, Tcool_t, RPM_t=None):
        """
        Dual-channel geometry exists, but flow goes through only ONE channel.
        -> Use the flowing channel (motor channel) to compute h.
        -> Do NOT downscale by area fraction here; handle area in the cooling term (h*A*ΔT).
        """

        device = LPM_t.device
        dtype  = LPM_t.dtype

        LPM = LPM_t.view(-1).clamp(min=0.0)
        _   = Tcool_t.view(-1)  # interface only

        # ---- 0) coolant ON/OFF ----
        eps = 1e-6
        coolant_on = (LPM > eps).to(dtype=dtype)  # [N]

        # ---- 1) total flow rate Q [m^3/s] ----
        Q_total = LPM * (1e-3 / 60.0)  # [N]

        # ---- 2) FLOW: all flow goes to the active channel ----
        Q_m = Q_total  # ✅ 핵심: 유량 분기 없음

        # ---- 3) channel geometry (motor channel) ----
        # NOTE: 직사각형 12mm x 8mm면 Dh ≈ 9.6e-3, Aflow ≈ 9.6e-5
        Dh_m = torch.tensor(float(getattr(self, "Dh_motor", 9.6e-3)), device=device, dtype=dtype)      # [m]
        Aflow_m = torch.tensor(float(getattr(self, "Aflow_motor", 9.6e-5)), device=device, dtype=dtype)  # [m^2]

        # ---- 4) coolant properties ----
        rho = torch.tensor(float(getattr(self, "rho_cool", 997.0)), device=device, dtype=dtype)     # kg/m3
        mu  = torch.tensor(float(getattr(self, "mu_cool", 0.00089)), device=device, dtype=dtype)    # Pa*s
        kf  = torch.tensor(float(getattr(self, "k_cool", 0.60)), device=device, dtype=dtype)        # W/mK
        cp  = torch.tensor(float(getattr(self, "cp_cool", 4180.0)), device=device, dtype=dtype)     # J/kgK
        Pr  = (cp * mu / (kf + 1e-12)).clamp(min=1e-6)

        # ---- 5) velocity -> Re -> Nu -> h ----
        Vm = (Q_m / (Aflow_m + 1e-12)).clamp(min=0.0)  # [N]
        Re_m = (rho * Vm * Dh_m / (mu + 1e-12)).clamp(min=0.0)

        def _Nu(Re):
            Nu_lam = torch.full_like(Re, 4.36)  # fully developed laminar, const q''
            Nu_tur = 0.023 * (Re.clamp(min=1.0) ** 0.8) * (Pr ** 0.4)  # Dittus-Boelter
            return torch.where(Re < 2300.0, Nu_lam, Nu_tur)

        Nu_m = _Nu(Re_m)
        h_m = (Nu_m * kf / (Dh_m + 1e-12)).clamp(min=0.0)  # [N]

        # ---- 6) global learnable scaling (optional) ----
        if hasattr(self, "log_alpha_h"):
            alpha = torch.exp(self.log_alpha_h.to(dtype))
            h_m = alpha * h_m

        # ---- 7) effective h ----
        # ✅ 유량이 흐르는 채널만 유효: h_eff = h_m
        h_eff = coolant_on * h_m

        return h_eff.view(-1, 1)



    def _loss_for_dataset(model, dataset):
        # 1) 손실 시퀀스 로딩
        loss_seq_tensor = load_and_prepare_sequence(dataset["loss_csv"], t_amb_fill=dataset.get("t_amb", None))
        model.set_T_amb(dataset.get("t_amb", 20.0))

        # 2) 프로파일 세팅 (기존 코드 유지)
        seq = loss_seq_tensor.squeeze(0).cpu().numpy()
        time_grid = np.arange(0, PREDICT_TIME + 1e-9, 1.0)
        model.set_q_profile(time_grid,
            denorm_channel(seq[:,0],'mag'),
            denorm_channel(seq[:,1],'rot'),
            denorm_channel(seq[:,3],'cop'),
            denorm_channel(seq[:,2],'sta'),
            LPM=denorm_channel(seq[:,4],'LPM')
  )

        # 3) GT 데이터 로딩
        df_gt = load_gt_with_time(dataset["gt_merged"])
        df_gt = df_gt[df_gt["time"] <= PREDICT_TIME].copy()

        t_train  = df_gt['time'].values
        Tpm_gt   = df_gt['pm_tmax'].values
        Tcoil_gt = df_gt['coil_tmax'].values

        # 4) 모델 손실 계산 (데이터 MSE 포함)
        loss, dbg = model.sequential_consistency_loss(
            loss_seq_tensor,
            time_s_np=t_train,
            Tpm_gt=Tpm_gt,
            Tch_gt=Tcoil_gt,
            # domain_id=dom_id_t,  # (참고) 모델에 따라 필요하다면 주석 해제해서 넣으세요
            
            λ_ic=1.0, 
            λ_seq=1.0, 
            
            # [수정 1] 실제 데이터(GT)와 맞추도록 강제 (기존 0.0 -> 50.0 이상)
            λ_data=1.0,   
            
            # [수정 2] 잘 학습된 Rollout(파란선)을 신경망이 따라하게 만듦 (새로 추가)
            λ_teacher=5.0, 
            teacher_stride=10 
        )
        print("\n[DBG] dataset name:", dataset["name"])
        print("[DBG] gt path:", dataset["gt_merged"])
        print("[DBG] gt head coil_tmax:", df_gt["coil_tmax"].head().to_list() if "coil_tmax" in df_gt else "NO coil_tmax")
        print("[DBG] gt coil_tmax min/max:", (df_gt["coil_tmax"].min(), df_gt["coil_tmax"].max()) if "coil_tmax" in df_gt else "NO coil_tmax")
        print("[DBG] gt time unique:", sorted(df_gt["time"].unique().tolist())[:10] if "time" in df_gt else "NO time")


        return loss, dbg



    # profiles 세팅
    def set_q_profile(self, time_s, q_mag, q_rot, q_cop, q_sta,
                    h=None, LPM=None, RPM=None):
        self.q_profile = (np.asarray(time_s), np.asarray(q_mag),
                        np.asarray(q_rot), np.asarray(q_cop), np.asarray(q_sta))
        self.h_profile   = (np.asarray(time_s), np.asarray(h))   if h   is not None else None
        self.LPM_profile = (np.asarray(time_s), np.asarray(LPM)) if LPM is not None else None
        self.RPM_profile = (np.asarray(time_s), np.asarray(RPM)) if RPM is not None else None


    def _interp_q_at(self, t_norm_vec):
        ts, qm, qr, qc, qs = self.q_profile
        t_phys = (t_norm_vec.detach().cpu().numpy().reshape(-1) * PREDICT_TIME)

        qm_t = np.interp(t_phys, ts, qm)
        qr_t = np.interp(t_phys, ts, qr)
        qc_t = np.interp(t_phys, ts, qc)
        qs_t = np.interp(t_phys, ts, qs)

        # h
        h_t = None
        if self.h_profile is not None:
            th, h_raw = self.h_profile
            h_t = np.interp(t_phys, th, h_raw)

        # LPM
        LPM_t = None
        if self.LPM_profile is not None:
            tl, LPM_raw = self.LPM_profile
            LPM_t = np.interp(t_phys, tl, LPM_raw)

        # RPM
        RPM_t = None
        if self.RPM_profile is not None:
            tr, RPM_raw = self.RPM_profile
            RPM_t = np.interp(t_phys, tr, RPM_raw)

        dev = t_norm_vec.device
        qm_t = torch.as_tensor(qm_t, dtype=torch.float64, device=dev)
        qr_t = torch.as_tensor(qr_t, dtype=torch.float64, device=dev)
        qc_t = torch.as_tensor(qc_t, dtype=torch.float64, device=dev)
        qs_t = torch.as_tensor(qs_t, dtype=torch.float64, device=dev)

        h_t   = None if h_t   is None else torch.as_tensor(h_t,   dtype=torch.float64, device=dev)
        LPM_t = None if LPM_t is None else torch.as_tensor(LPM_t, dtype=torch.float64, device=dev)
        RPM_t = None if RPM_t is None else torch.as_tensor(RPM_t, dtype=torch.float64, device=dev)

        # Tamb는 고정 20 → 호출부에서 텐서로 만들어 쓰도록 None 반환
        Tamb_t = None

        return qm_t, qr_t, qc_t, qs_t, h_t, LPM_t, Tamb_t, RPM_t



    # DeepONet 출력 결합
    # [PATCH 2] --- DeepONet output 결합 수정 ---
    def output(self, u, t_feat):
        """
        u:      [N, C] (시간별 branch) 또는 [1, C] (상수 계수)
        t_feat: [N, C] (Trunk(t) 출력)
        return: [N, 1]
        """
        assert u.dim()==2 and t_feat.dim()==2, f"2D only: u={u.shape}, t_feat={t_feat.shape}"

        # 같은 시각끼리만 곱하기
        if u.size(0) == t_feat.size(0):
            return (u * t_feat).sum(dim=1, keepdim=True)

        # u가 상수 계수라면 N으로 반복해 '같은 t' 규칙 유지
        if u.size(0) == 1:
            u_rep = u.expand(t_feat.size(0), -1)           # [N, C]
            return (u_rep * t_feat).sum(dim=1, keepdim=True)

        # 그 외는 명시적으로 에러
        raise ValueError(f"Time alignment mismatch: u={u.size()}, t_feat={t_feat.size()}")


    

        # === ΔT 모드 헬퍼 ===
    def _Tamb0(self) -> torch.Tensor:
        """t=0 주변온도 스칼라"""
        if getattr(self, "Tamb_profile", None) is not None:
            val = float(np.interp(0.0, self.Tamb_profile[0], self.Tamb_profile[1]))
            return torch.tensor(val, dtype=torch.float64, device=self.device)
        return torch.tensor(float(self.T_amb), dtype=torch.float64, device=self.device)

    def _Tamb_at_vec(self, t_vec01: torch.Tensor) -> torch.Tensor:
        """정규화 t(0~1) 벡터에서 Tamb(t) (없으면 상수)"""
        if getattr(self, "Tamb_profile", None) is None:
            return self.T_amb.view(1).expand(t_vec01.shape[0])
        t_phys = (t_vec01.view(-1).detach().cpu().numpy() * PREDICT_TIME)
        vals = np.interp(t_phys, self.Tamb_profile[0], self.Tamb_profile[1])
        return torch.as_tensor(vals, dtype=torch.float64, device=self.device)

    def _dTamb_vec(self, t_vec01: torch.Tensor) -> torch.Tensor:
        """ΔTamb(t) = Tamb(t) - Tamb0"""
        return self._Tamb_at_vec(t_vec01) - self._Tamb0()


    # ★MOD: t+controls → Trunk 전달
    # ★★★ 교체: 내부 상태를 ΔT로 해석

    # [PATCH 3] --- _node_temp_and_dt 주석+호출 일관화 ---
    def _node_temp_and_dt(self, bout, trunk, t_vec):
        """
        bout:  [N, C] (시간별 branch) 또는 [1, C] (상수 branch)
        trunk: t-only Trunk (예: self.PM, self.COIL_H 등)
        t_vec: [N, 1]  정규화된 시간 (requires_grad=True면 d/dt 가능)
        """
        tout = trunk(t_vec)                       # [N, C], t-only
        S = self.output(bout, tout).view(-1, 1)   # [N,1] (행별 내적/기존 matmul 둘 다 지원)
        dSdt = torch.autograd.grad(S.sum(), t_vec, create_graph=True)[0].view(-1, 1)
        dT_rel = S * T_MAX
        ddTdt  = dSdt * T_MAX / PREDICT_TIME
        return dT_rel, ddTdt


    # 이하 LPTN_residuals(), sequential_consistency_loss(), predict(), predict_rollout()
    # 및 학습/테스트 루프는 GRU9 기존 코드 그대로 유지
    # ---------------------------------------------





    def sequential_consistency_loss(
        self,
        loss_sequence,
        time_s_np,
        Tpm_gt=None,
        Tcs_gt=None,
        Tch_gt=None,
        domain_id=None,
        λ_ic=1,
        λ_seq=1.0,          # ✅ 권장: 5~20 (기존 100 → 낮춤)
        λ_data=1.0,
        gt_time_s_np=None,
        λ_teacher=5.0,     
        teacher_stride=5,
    ):
        """
        Hybrid loss:
        - Physics residual (autograd dT/dt): loss_phy
        - IC: ΔT(0)=0 : loss_ic
        - Sequential consistency (Euler 1-step): seq_loss
        - Sparse data loss at GT timestamps (optional): data_loss
        - Smoothness regularizer (optional): L_smooth
        - Teacher-rollout distillation (optional): L_teacher  (direct -> rollout)
        """
        import numpy as np
        import torch

        model_dtype = next(self.parameters()).dtype
        device = self.device

        # -------------------------
        # helpers
        # -------------------------
        def _v(x):
            if x is None:
                return None
            x = x if torch.is_tensor(x) else torch.as_tensor(x, device=device)
            x = x.to(device=device, dtype=model_dtype)
            if x.dim() == 1:
                return x.view(-1, 1)
            return x

        def _to_scalar(x, reduce="mean"):
            if torch.is_tensor(x):
                if x.numel() == 1:
                    return x.reshape(())
                if reduce == "mean":
                    return x.mean()
                if reduce == "sum":
                    return x.sum()
            return torch.as_tensor(x, dtype=model_dtype, device=device).reshape(())

        def _clamp_R(R):
            return torch.clamp(R, min=1e-9)

        # ============================================================
        # 1) Branch: GRU -> BN
        # ============================================================
        h_seq = self.gru_encoder(loss_sequence, domain_id=domain_id, return_sequences=True).squeeze(0)  # [Nh_full,H]
        Nh_full = int(h_seq.size(0))
        u_seq_full = h_seq

        # ============================================================
        # 2) time axis  (✅ dt_base 기반 정렬로 변경)
        # ============================================================
        t_np = np.asarray(time_s_np, dtype=np.float64).reshape(-1)
        t_np = np.unique(t_np)
        t_np.sort()
        Nt = int(len(t_np))

        if Nh_full < 2:
            idx = np.zeros((Nt,), dtype=np.int64)
        else:
            dt_base = float(PREDICT_TIME) / float(Nh_full - 1)  # ✅ 1초/10초 등 자동 대응
            idx = np.rint(t_np / dt_base).astype(np.int64)
            idx = np.clip(idx, 0, Nh_full - 1)

        u_seq = u_seq_full[idx]          # [Nt, H]
        bout  = self.BN(u_seq)           # [Nt, Cb]

        N = min(len(t_np), int(bout.size(0)))
        t_np = t_np[:N]
        bout = bout[:N]
        u_seq = u_seq[:N]

        # normalized time [0,1]
        t = torch.as_tensor(
            np.clip(t_np / PREDICT_TIME, 0.0, 1.0),
            dtype=model_dtype, device=device
        ).view(-1, 1)
        t.requires_grad_(True)

        # ============================================================
        # 3) Trunk: basis
        # ============================================================
        phi_pm = self.PM(t).to(model_dtype)

        if hasattr(self, "ROTOR_A"):
            phi_ra = self.ROTOR_A(t).to(model_dtype)
        else:
            phi_ra = (self.ROTOR(t) if hasattr(self, "ROTOR") else self.PM(t)).to(model_dtype)

        if hasattr(self, "ROTOR_B"):
            phi_rb = self.ROTOR_B(t).to(model_dtype)
        else:
            phi_rb = (self.ROTOR(t) if hasattr(self, "ROTOR") else self.PM(t)).to(model_dtype)

        phi_cs  = self.CS(t).to(model_dtype)
        phi_st  = self.STATOR(t).to(model_dtype)
        phi_end = self.END(t).to(model_dtype)

        if hasattr(self, "HOUSING"):
            phi_hou = self.HOUSING(t).to(model_dtype)
        else:
            phi_hou = self.STATOR(t).to(model_dtype)

        assert bout.size(0) == phi_pm.size(0) == phi_ra.size(0) == phi_rb.size(0) == phi_cs.size(0) == phi_st.size(0) == phi_end.size(0) == phi_hou.size(0) == N

        def dot(b, phi):
            return (b * phi).sum(dim=1, keepdim=True)

        S_pm  = dot(bout, phi_pm)
        S_ra  = dot(bout, phi_ra)
        S_rb  = dot(bout, phi_rb)
        S_cs  = dot(bout, phi_cs)
        S_st  = dot(bout, phi_st)
        S_end = dot(bout, phi_end)
        S_hou = dot(bout, phi_hou)

        dSdt_pm  = torch.autograd.grad(S_pm.sum(),  t, create_graph=True)[0]
        dSdt_ra  = torch.autograd.grad(S_ra.sum(),  t, create_graph=True)[0]
        dSdt_rb  = torch.autograd.grad(S_rb.sum(),  t, create_graph=True)[0]
        dSdt_cs  = torch.autograd.grad(S_cs.sum(),  t, create_graph=True)[0]
        dSdt_st  = torch.autograd.grad(S_st.sum(),  t, create_graph=True)[0]
        dSdt_end = torch.autograd.grad(S_end.sum(), t, create_graph=True)[0]
        dSdt_hou = torch.autograd.grad(S_hou.sum(), t, create_graph=True)[0]

        dTpm  = S_pm  * T_MAX
        dTra  = S_ra  * T_MAX
        dTrb  = S_rb  * T_MAX
        dTcs  = S_cs  * T_MAX
        dTst  = S_st  * T_MAX
        dTend = S_end * T_MAX
        dTh   = S_hou * T_MAX

        scale_dt = (T_MAX / PREDICT_TIME)
        dTpm_dt  = dSdt_pm  * scale_dt
        dTra_dt  = dSdt_ra  * scale_dt
        dTrb_dt  = dSdt_rb  * scale_dt
        dTcs_dt  = dSdt_cs  * scale_dt
        dTst_dt  = dSdt_st  * scale_dt
        dTend_dt = dSdt_end * scale_dt
        dTh_dt   = dSdt_hou * scale_dt

        # ============================================================
        # 4) Interp q/h/Tamb/RPM
        # ============================================================
        q_mag, q_rot, q_cop, q_sta, h_t, LPM_t, Tamb_t, RPM_t = self._interp_q_at(t.view(-1))
        q_mag = _v(q_mag); q_rot = _v(q_rot); q_cop = _v(q_cop); q_sta = _v(q_sta)
        h_t   = _v(h_t);   LPM_t = _v(LPM_t); Tamb_t = _v(Tamb_t); RPM_t = _v(RPM_t)

        if Tamb_t is not None:
            T_amb_vec = Tamb_t
        else:
            if torch.is_tensor(self.T_amb):
                T_amb_vec = self.T_amb.to(device=device, dtype=model_dtype).view(1, 1).expand(N, 1)
            else:
                T_amb_vec = torch.full((N, 1), float(self.T_amb), dtype=model_dtype, device=device)

        if RPM_t is None:
            RPM_t = torch.zeros_like(T_amb_vec)
        if LPM_t is None:
            LPM_t = torch.zeros_like(T_amb_vec)

        T_cool0 = 30.0
        T_cool_vec = torch.full((N, 1), float(T_cool0), dtype=model_dtype, device=device)

        # ============================================================
        # 5) Absolute temperatures
        # ============================================================
        T_pm  = T_amb_vec + dTpm
        T_ra  = T_amb_vec + dTra
        T_rb  = T_amb_vec + dTrb
        T_cs  = T_amb_vec + dTcs
        T_st  = T_amb_vec + dTst
        T_end = T_amb_vec + dTend
        T_hou = T_amb_vec + dTh

        # ============================================================
        # 5.5) Copper loss correction + split
        # ============================================================
        alpha_cu = self.alpha_cu.to(model_dtype) if hasattr(self, "alpha_cu") else torch.tensor(0.0, device=device, dtype=model_dtype)
        T_ref_vec = T_amb_vec
        factor = 1.0 + alpha_cu * (T_cs - T_ref_vec)
        factor = torch.clamp(factor, min=0.7, max=2.0)
        q_cop_eff = q_cop * factor

        V_cs  = 4.899e-4
        V_end = 9.024e-4
        f_cs  = V_cs  / (V_cs + V_end)
        f_end = V_end / (V_cs + V_end)
        q_cop_cs  = q_cop_eff * f_cs
        q_cop_end = q_cop_eff * f_end

        if hasattr(self, "V_rotorA") and hasattr(self, "V_rotorB"):
            Va = float(self.V_rotorA)
            Vb = float(self.V_rotorB)
            fr_a = Va / (Va + Vb + 1e-12)
        else:
            fr_a = 0.5
        q_rot_a = q_rot * fr_a
        q_rot_b = q_rot * (1.0 - fr_a)

        Tcool0 = 30.0  # coolant temperature [°C], constant
        Tcool0 = float(Tcool0)


        # ============================================================
        # 6) Dynamic water h + gates + air convection
        # ============================================================
        if getattr(self, "use_dynamic_h", False):
            if h_t is not None:
                h_dyn = torch.clamp(h_t, min=1e-6)
            else:
                # Tcool_vec: [N,1] already defined above
                h_dyn = self._h_from_LPM_tcool_rpm_dual(
                    LPM_t.view(-1),          # [N]
                    T_cool_vec.view(-1),     # [N]
                    RPM_t.view(-1)           # [N]
                )  # -> [N,1]
                h_dyn = torch.clamp(h_dyn, min=1e-6)


        else:
            h_dyn = None

        LPM_EPS = 1e-6
        coolant_on = (LPM_t > LPM_EPS).to(dtype=model_dtype)

        R_st_cool_t = None
        if h_dyn is not None:
            if hasattr(self, "A_st_ch"):
                A_st_use = self.A_st_ch.to(device=device, dtype=model_dtype)
            elif hasattr(self, "A_st"):
                A_st_use = self.A_st.to(device=device, dtype=model_dtype)
            else:
                A_st_use = None

            if A_st_use is not None:
                R_conv = 1.0 / (h_dyn * A_st_use + 1e-9)
                if hasattr(self, "R_st_ch"):
                    R_wall = self.R_st_ch.to(device=device, dtype=model_dtype)
                    R_st_cool_raw = _clamp_R(R_wall + R_conv)
                else:
                    R_st_cool_raw = _clamp_R(R_conv)
                R_st_cool_t = R_st_cool_raw

        h_air = self.h_air.to(device=device, dtype=model_dtype) if hasattr(self, "h_air") else torch.tensor(20.0, device=device, dtype=model_dtype)
        A_end_surf = self.A_end_surf.to(device=device, dtype=model_dtype) if hasattr(self, "A_end_surf") else torch.tensor(0.01023, device=device, dtype=model_dtype)
        A_pm_surf  = self.A_pm_surf.to(device=device, dtype=model_dtype) if hasattr(self, "A_pm_surf") else torch.tensor(0.00156, device=device, dtype=model_dtype)
        R_end_amb = _clamp_R(1.0 / (h_air * A_end_surf + 1e-9))
        R_pm_amb  = _clamp_R(1.0 / (h_air * A_pm_surf  + 1e-9))

        # ============================================================
        # 6.5) Air-gaps: RB-CS and RA-CS
        # ============================================================
        k_air = self.k_air.to(device=device, dtype=model_dtype)
        delta_gap = self.delta_gap.to(device=device, dtype=model_dtype)
        h_cond = k_air / (delta_gap + 1e-12)

        h_conv0 = self.h_conv0.to(device=device, dtype=model_dtype)
        h_conv_exp = self.h_conv_exp.to(device=device, dtype=model_dtype)
        rpm_ref = self.RPM_NORM_REF.to(device=device, dtype=model_dtype)

        rpm_n = RPM_t / (rpm_ref + 1e-12)
        h_conv_rot = h_conv0 * torch.clamp(rpm_n, min=0.0) ** h_conv_exp
        h_gap_eff = h_cond + h_conv_rot

        A_rb_cs = self.A_rb_cs.to(device=device, dtype=model_dtype)
        R_rb_cs_t = _clamp_R(1.0 / (h_gap_eff * A_rb_cs + 1e-9))

        if hasattr(self, "A_ra_cs"):
            A_ra_cs = self.A_ra_cs.to(device=device, dtype=model_dtype)
        else:
            A_ra_cs = A_rb_cs
        R_ra_cs_t = _clamp_R(1.0 / (h_gap_eff * A_ra_cs + 1e-9))

        # ============================================================
        # 7) Constants / resistances
        # ============================================================
        C_pm  = self.C_pm.to(device=device, dtype=model_dtype)
        C_ra  = (self.C_rotorA if hasattr(self, "C_rotorA") else self.C_rotor).to(device=device, dtype=model_dtype)
        C_rb  = (self.C_rotorB if hasattr(self, "C_rotorB") else self.C_rotor).to(device=device, dtype=model_dtype)
        C_cs  = self.C_CS.to(device=device, dtype=model_dtype)
        C_st  = self.C_stator.to(device=device, dtype=model_dtype)
        C_end = self.C_end.to(device=device, dtype=model_dtype)
        C_hou = (self.C_housing if hasattr(self, "C_housing") else torch.tensor(28823.0319, device=device, dtype=model_dtype)).to(device=device, dtype=model_dtype)

        R_pm_ra  = (self.R_ra_pm if hasattr(self, "R_ra_pm") else self.R_rt_pm).to(device=device, dtype=model_dtype)
        R_pm_rb  = (self.R_pm_rb if hasattr(self, "R_pm_rb") else self.R_rt_pm).to(device=device, dtype=model_dtype)
        R_cs_st  = (self.R_cs_st if hasattr(self, "R_cs_st") else self.R_ch_st).to(device=device, dtype=model_dtype)
        R_cs_end = self.R_cs_end.to(device=device, dtype=model_dtype)

        R_st_hou  = (self.R_st_hou  if hasattr(self, "R_st_hou")  else torch.tensor(2.0,   device=device, dtype=model_dtype)).to(device=device, dtype=model_dtype)
        R_end_hou = (self.R_end_hou if hasattr(self, "R_end_hou") else torch.tensor(0.126, device=device, dtype=model_dtype)).to(device=device, dtype=model_dtype)
        R_hou_amb = (self.R_hou_amb if hasattr(self, "R_hou_amb") else torch.tensor(1000.0,device=device, dtype=model_dtype)).to(device=device, dtype=model_dtype)

        R_pm_ra   = _clamp_R(R_pm_ra)
        R_pm_rb   = _clamp_R(R_pm_rb)
        R_cs_st   = _clamp_R(R_cs_st)
        R_cs_end  = _clamp_R(R_cs_end)
        R_st_hou  = _clamp_R(R_st_hou)
        R_end_hou = _clamp_R(R_end_hou)
        R_hou_amb = _clamp_R(R_hou_amb)

        # ============================================================
        # 8) Physics residuals (LPTN ODE residual)
        # ============================================================
        res_pm = (
            C_pm * dTpm_dt
            - q_mag
            + (T_pm - T_ra) / R_pm_ra
            + (T_pm - T_rb) / R_pm_rb
            + (T_pm - T_amb_vec) / R_pm_amb
        )
        res_ra = (
            C_ra * dTra_dt
            - q_rot_a
            + (T_ra - T_pm) / R_pm_ra
            + (T_ra - T_cs) / R_ra_cs_t
        )
        res_rb = (
            C_rb * dTrb_dt
            - q_rot_b
            + (T_rb - T_pm) / R_pm_rb
            + (T_rb - T_cs) / R_rb_cs_t
        )
        res_cs = (
            C_cs * dTcs_dt
            - q_cop_cs
            + (T_cs - T_ra) / R_ra_cs_t
            + (T_cs - T_rb) / R_rb_cs_t
            + (T_cs - T_st) / R_cs_st
            + (T_cs - T_end) / R_cs_end
        )
        res_st = (
            C_st * dTst_dt
            - q_sta
            + (T_st - T_cs) / R_cs_st
            + (T_st - T_hou) / R_st_hou
        )
        if R_st_cool_t is not None:
            res_st = res_st + coolant_on * (T_st - T_cool_vec) / R_st_cool_t

        res_end = (
            C_end * dTend_dt
            - q_cop_end
            + (T_end - T_cs) / R_cs_end
            + (T_end - T_amb_vec) / R_end_amb
            + (T_end - T_hou) / R_end_hou
        )
        res_hou = (
            C_hou * dTh_dt
            - (T_st - T_hou) / R_st_hou
            - (T_end - T_hou) / R_end_hou
            + (T_hou - T_amb_vec) / R_hou_amb
        )

        loss_pm  = (res_pm**2).mean()  / 1e7
        loss_ra  = (res_ra**2).mean()  / 1e6
        loss_rb  = (res_rb**2).mean()  / 1e5
        loss_cs  = (res_cs**2).mean()  / 1e7
        loss_st  = (res_st**2).mean()  / 1e8
        loss_end = (res_end**2).mean() / 1e5
        loss_hou = (res_hou**2).mean() / 1e6
        loss_phy = loss_pm + loss_ra + loss_rb + loss_cs + loss_st + loss_end + loss_hou

        # ============================================================
        # 9) IC: ΔT(0)=0 for all nodes
        # ============================================================
        t0 = torch.zeros((1, 1), dtype=model_dtype, device=device)
        b0 = bout[0:1]

        def S0(phi_fn):
            return (b0 * phi_fn(t0).to(model_dtype)).sum(dim=1, keepdim=True)

        S0_pm  = S0(self.PM)
        S0_ra  = S0(self.ROTOR_A) if hasattr(self, "ROTOR_A") else (S0(self.ROTOR) if hasattr(self, "ROTOR") else S0(self.PM))
        S0_rb  = S0(self.ROTOR_B) if hasattr(self, "ROTOR_B") else (S0(self.ROTOR) if hasattr(self, "ROTOR") else S0(self.PM))
        S0_cs  = S0(self.CS)
        S0_st  = S0(self.STATOR)
        S0_end = S0(self.END)
        S0_hou = S0(self.HOUSING) if hasattr(self, "HOUSING") else S0(self.STATOR)

        dT0 = (
            (S0_pm.squeeze()*T_MAX).pow(2)
            + (S0_ra.squeeze()*T_MAX).pow(2)
            + (S0_rb.squeeze()*T_MAX).pow(2)
            + (S0_cs.squeeze()*T_MAX).pow(2)
            + (S0_st.squeeze()*T_MAX).pow(2)
            + (S0_end.squeeze()*T_MAX).pow(2)
            + (S0_hou.squeeze()*T_MAX).pow(2)
        )
        loss_ic = (λ_ic * dT0)

        # ============================================================
        # 10) Smoothness regularizer
        # ============================================================
        lambda_smooth   = getattr(self, 'lambda_smooth', 1e-3)
        kappa_smooth    = getattr(self, 'kappa_smooth', 5.0)
        use_gate_smooth = getattr(self, 'use_gate_smooth', True)

        if N < 2:
            L_smooth = torch.zeros((), dtype=model_dtype, device=device)
        else:
            Cb = bout.size(1)
            c_diff = bout[1:] - bout[:-1]
            per_step_energy = (c_diff.pow(2).sum(dim=1) / (Cb + 1e-8))

            if use_gate_smooth:
                du = u_seq[1:] - u_seq[:-1]
                du_norm = du.norm(dim=1)
                z = kappa_smooth * du_norm
                gate = 1.0 - (z / (1.0 + z))
                gate = gate.detach()
            else:
                gate = torch.ones_like(per_step_energy)

            Lm = min(gate.size(0), per_step_energy.size(0))
            L_smooth = lambda_smooth * (gate[:Lm] * per_step_energy[:Lm]).mean()

        # ============================================================
        # 11) Data loss (sparse GT)
        # ============================================================
        if (Tcs_gt is None) and (Tch_gt is not None):
            Tcs_gt = Tch_gt

            

        data_loss = torch.zeros((), dtype=model_dtype, device=device)

        def _sparse_mse(T_pred_1d, gt_times_np, gt_vals_np):
            if gt_times_np is None or gt_vals_np is None:
                return None
            if len(gt_times_np) == 0:
                return None

            t_all = torch.as_tensor(t_np, dtype=model_dtype, device=device).view(-1)  # [N]
            gt_t  = torch.as_tensor(gt_times_np, dtype=model_dtype, device=device).view(-1)
            gt_v  = torch.as_tensor(gt_vals_np,  dtype=model_dtype, device=device).view(-1)

            valid = torch.isfinite(gt_t) & torch.isfinite(gt_v)
            if valid.sum().item() == 0:
                return None
            gt_t = gt_t[valid]
            gt_v = gt_v[valid]

            idx_hi = torch.searchsorted(t_all, gt_t)
            idx_hi = torch.clamp(idx_hi, 0, t_all.numel() - 1)
            idx_lo = torch.clamp(idx_hi - 1, 0, t_all.numel() - 1)

            dist_hi = torch.abs(t_all[idx_hi] - gt_t)
            dist_lo = torch.abs(t_all[idx_lo] - gt_t)
            idx = torch.where(dist_lo <= dist_hi, idx_lo, idx_hi)

            pred = T_pred_1d[idx]
            return ((pred - gt_v) ** 2).mean()

        if gt_time_s_np is not None:
            if Tpm_gt is not None:
                mse_pm = _sparse_mse(T_pm.view(-1), gt_time_s_np, Tpm_gt)
                if mse_pm is not None:
                    data_loss = data_loss + mse_pm /0.2e3
            if Tcs_gt is not None:
                mse_cs = _sparse_mse(T_cs.view(-1), gt_time_s_np, Tcs_gt)
                if mse_cs is not None:
                    data_loss = data_loss + mse_cs /0.2e3
        else:
            if Tpm_gt is not None and len(Tpm_gt) == N:
                Tpm_gt_t = torch.as_tensor(Tpm_gt, dtype=model_dtype, device=device).view(-1, 1)[:N]
                data_loss = (data_loss + ((T_pm - Tpm_gt_t) ** 2).mean())/0.2e3
            if Tcs_gt is not None and len(Tcs_gt) == N:
                Tcs_gt_t = torch.as_tensor(Tcs_gt, dtype=model_dtype, device=device).view(-1, 1)[:N]
                data_loss = (data_loss + ((T_cs - Tcs_gt_t) ** 2).mean())/0.2e3

        # ============================================================
        # 12) Sequential consistency (Euler 1-step)
        # ============================================================
        if N < 2:
            seq_loss = torch.zeros((), dtype=model_dtype, device=device)
        else:
            dts = torch.as_tensor(np.diff(t_np), dtype=model_dtype, device=device).view(-1, 1)
            dts = torch.clamp(dts, min=0.0)

            Tpm_k  = T_pm[:-1];  Tra_k  = T_ra[:-1];  Trb_k  = T_rb[:-1]
            Tcs_k  = T_cs[:-1];  Tst_k  = T_st[:-1];  Tend_k = T_end[:-1];  Thou_k = T_hou[:-1]
            Tamb_k = T_amb_vec[:-1]

            qmag_k = q_mag[:-1]; qsta_k = q_sta[:-1]
            qrot_a_k = q_rot_a[:-1]; qrot_b_k = q_rot_b[:-1]
            qcop_cs_k  = q_cop_cs[:-1]; qcop_end_k = q_cop_end[:-1]

            Rra_cs_k = R_ra_cs_t[:-1]
            Rrb_cs_k = R_rb_cs_t[:-1]

            if R_st_cool_t is not None:
                Rst_cool_k = R_st_cool_t[:-1]
                cool_on_k  = coolant_on[:-1]
                Tcool_k    = T_cool_vec[:-1]
            else:
                Rst_cool_k = None

            dTpm_rhs = (
                qmag_k
                - (Tpm_k - Tra_k) / R_pm_ra
                - (Tpm_k - Trb_k) / R_pm_rb
                - (Tpm_k - Tamb_k) / R_pm_amb
            ) / C_pm

            dTra_rhs = (
                qrot_a_k
                - (Tra_k - Tpm_k) / R_pm_ra
                - (Tra_k - Tcs_k) / Rra_cs_k
            ) / C_ra

            dTrb_rhs = (
                qrot_b_k
                - (Trb_k - Tpm_k) / R_pm_rb
                - (Trb_k - Tcs_k) / Rrb_cs_k
            ) / C_rb

            dTcs_rhs = (
                qcop_cs_k
                - (Tcs_k - Tra_k) / Rra_cs_k
                - (Tcs_k - Trb_k) / Rrb_cs_k
                - (Tcs_k - Tst_k) / R_cs_st
                - (Tcs_k - Tend_k) / R_cs_end
            ) / C_cs

            dTst_rhs = (
                qsta_k
                - (Tst_k - Tcs_k) / R_cs_st
                - (Tst_k - Thou_k) / R_st_hou
            )
            if Rst_cool_k is not None:
                dTst_rhs = dTst_rhs - cool_on_k * (Tst_k - Tcool_k) / Rst_cool_k
            dTst_rhs = dTst_rhs / C_st

            dTend_rhs = (
                qcop_end_k
                - (Tend_k - Tcs_k) / R_cs_end
                - (Tend_k - Tamb_k) / R_end_amb
                - (Tend_k - Thou_k) / R_end_hou
            ) / C_end

            dThou_rhs = (
                (Tst_k - Thou_k) / R_st_hou
                + (Tend_k - Thou_k) / R_end_hou
                - (Thou_k - Tamb_k) / R_hou_amb
            ) / C_hou

            Tpm_e  = Tpm_k  + dts * dTpm_rhs
            Tra_e  = Tra_k  + dts * dTra_rhs
            Trb_e  = Trb_k  + dts * dTrb_rhs
            Tcs_e  = Tcs_k  + dts * dTcs_rhs
            Tst_e  = Tst_k  + dts * dTst_rhs
            Tend_e = Tend_k + dts * dTend_rhs
            Thou_e = Thou_k + dts * dThou_rhs

            seq_loss = (
                ((Tpm_e  - T_pm[1:])  ** 2).mean()
                + ((Tra_e  - T_ra[1:])  ** 2).mean()
                + ((Trb_e  - T_rb[1:])  ** 2).mean()
                + ((Tcs_e  - T_cs[1:])  ** 2).mean()
                + ((Tst_e  - T_st[1:])  ** 2).mean()
                + ((Tend_e - T_end[1:]) ** 2).mean()
                + ((Thou_e - T_hou[1:]) ** 2).mean()
            )

        # ============================================================
        # 12.5) Teacher-rollout distillation (✅ teacher IC에서 시작하도록 수정)
        # ============================================================
        L_teacher = torch.zeros((), dtype=model_dtype, device=device)
        if (λ_teacher is not None) and (float(λ_teacher) > 0.0) and (N >= 2):
            with torch.no_grad():
                stride = int(teacher_stride) if teacher_stride is not None else 1
                stride = max(1, stride)

                idx_use = torch.arange(0, N, stride, device=device)
                if idx_use[-1].item() != (N - 1):
                    idx_use = torch.cat([idx_use, torch.tensor([N - 1], device=device)], dim=0)

                t_np_use = t_np[idx_use.detach().cpu().numpy()]
                dts_use = torch.as_tensor(np.diff(t_np_use), dtype=model_dtype, device=device).view(-1, 1)
                dts_use = torch.clamp(dts_use, min=0.0)

                # subsampled exogenous / parameters
                qmag_u   = q_mag[idx_use]
                qsta_u   = q_sta[idx_use]
                Tamb_u   = T_amb_vec[idx_use]
                Tcool_u  = T_cool_vec[idx_use]
                coolon_u = coolant_on[idx_use]

                qrota_u   = q_rot_a[idx_use]
                qrotb_u   = q_rot_b[idx_use]
                qcopcs_u  = q_cop_cs[idx_use]
                qcopend_u = q_cop_end[idx_use]

                Rra_cs_u   = R_ra_cs_t[idx_use]
                Rrb_cs_u   = R_rb_cs_t[idx_use]
                Rstcool_u  = (R_st_cool_t[idx_use] if (R_st_cool_t is not None) else None)

                # ✅ teacher init from IC (Tamb at start), NOT from direct
                Tamb0 = float(T_amb_vec[0].detach().item())
                Tpm_r  = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)
                Tra_r  = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)
                Trb_r  = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)
                Tcs_r  = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)
                Tst_r  = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)
                Tend_r = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)
                Thou_r = torch.tensor([[Tamb0]], dtype=model_dtype, device=device)

                Tpm_roll  = [Tpm_r]
                Tra_roll  = [Tra_r]
                Trb_roll  = [Trb_r]
                Tcs_roll  = [Tcs_r]
                Tst_roll  = [Tst_r]
                Tend_roll = [Tend_r]
                Thou_roll = [Thou_r]

                for k in range(dts_use.size(0)):
                    dt = dts_use[k:k+1]

                    qmag = qmag_u[k:k+1]
                    qsta = qsta_u[k:k+1]
                    Tamb = Tamb_u[k:k+1]

                    qrota   = qrota_u[k:k+1]
                    qrotb   = qrotb_u[k:k+1]
                    qcopcs  = qcopcs_u[k:k+1]
                    qcopend = qcopend_u[k:k+1]

                    Rra_cs = Rra_cs_u[k:k+1]
                    Rrb_cs = Rrb_cs_u[k:k+1]

                    dTpm_rhs = (
                        qmag
                        - (Tpm_r - Tra_r) / R_pm_ra
                        - (Tpm_r - Trb_r) / R_pm_rb
                        - (Tpm_r - Tamb)  / R_pm_amb
                    ) / C_pm

                    dTra_rhs = (
                        qrota
                        - (Tra_r - Tpm_r) / R_pm_ra
                        - (Tra_r - Tcs_r) / Rra_cs
                    ) / C_ra

                    dTrb_rhs = (
                        qrotb
                        - (Trb_r - Tpm_r) / R_pm_rb
                        - (Trb_r - Tcs_r) / Rrb_cs
                    ) / C_rb

                    dTcs_rhs = (
                        qcopcs
                        - (Tcs_r - Tra_r) / Rra_cs
                        - (Tcs_r - Trb_r) / Rrb_cs
                        - (Tcs_r - Tst_r) / R_cs_st
                        - (Tcs_r - Tend_r)/ R_cs_end
                    ) / C_cs

                    dTst_rhs = (
                        qsta
                        - (Tst_r - Tcs_r) / R_cs_st
                        - (Tst_r - Thou_r)/ R_st_hou
                    )
                    if Rstcool_u is not None:
                        Rstc   = Rstcool_u[k:k+1]
                        coolon = coolon_u[k:k+1]
                        Tcool  = Tcool_u[k:k+1]
                        dTst_rhs = dTst_rhs - coolon * (Tst_r - Tcool) / Rstc
                    dTst_rhs = dTst_rhs / C_st

                    dTend_rhs = (
                        qcopend
                        - (Tend_r - Tcs_r) / R_cs_end
                        - (Tend_r - Tamb)  / R_end_amb
                        - (Tend_r - Thou_r)/ R_end_hou
                    ) / C_end

                    dThou_rhs = (
                        (Tst_r - Thou_r) / R_st_hou
                        + (Tend_r - Thou_r) / R_end_hou
                        - (Thou_r - Tamb) / R_hou_amb
                    ) / C_hou

                    Tpm_r  = Tpm_r  + dt * dTpm_rhs
                    Tra_r  = Tra_r  + dt * dTra_rhs
                    Trb_r  = Trb_r  + dt * dTrb_rhs
                    Tcs_r  = Tcs_r  + dt * dTcs_rhs
                    Tst_r  = Tst_r  + dt * dTst_rhs
                    Tend_r = Tend_r + dt * dTend_rhs
                    Thou_r = Thou_r + dt * dThou_rhs

                    Tpm_roll.append(Tpm_r)
                    Tra_roll.append(Tra_r)
                    Trb_roll.append(Trb_r)
                    Tcs_roll.append(Tcs_r)
                    Tst_roll.append(Tst_r)
                    Tend_roll.append(Tend_r)
                    Thou_roll.append(Thou_r)

                Tpm_teacher  = torch.cat(Tpm_roll,  dim=0)
                Tra_teacher  = torch.cat(Tra_roll,  dim=0)
                Trb_teacher  = torch.cat(Trb_roll,  dim=0)
                Tcs_teacher  = torch.cat(Tcs_roll,  dim=0)
                Tst_teacher  = torch.cat(Tst_roll,  dim=0)
                Tend_teacher = torch.cat(Tend_roll, dim=0)
                Thou_teacher = torch.cat(Thou_roll, dim=0)

            # student direct (subsample)
            Tpm_direct_u  = T_pm[idx_use]
            Tra_direct_u  = T_ra[idx_use]
            Trb_direct_u  = T_rb[idx_use]
            Tcs_direct_u  = T_cs[idx_use]
            Tst_direct_u  = T_st[idx_use]
            Tend_direct_u = T_end[idx_use]
            Thou_direct_u = T_hou[idx_use]

            L_teacher = (
                ((Tpm_direct_u  - Tpm_teacher )**2).mean()
            + ((Tra_direct_u  - Tra_teacher )**2).mean()
            + ((Trb_direct_u  - Trb_teacher )**2).mean()
            + ((Tcs_direct_u  - Tcs_teacher )**2).mean()
            + ((Tst_direct_u  - Tst_teacher )**2).mean()
            + ((Tend_direct_u - Tend_teacher)**2).mean()
            + ((Thou_direct_u - Thou_teacher)**2).mean()
            )

        # ============================================================
        # 13) Total
        # ============================================================
        loss_phy_s  = _to_scalar(loss_phy, "mean")
        loss_ic_s   = _to_scalar(loss_ic, "mean")
        seq_loss_s  = _to_scalar(seq_loss, "mean")
        data_loss_s = _to_scalar(data_loss, "mean")
        L_smooth_s  = _to_scalar(L_smooth, "mean")
        L_teacher_s = _to_scalar(L_teacher, "mean")
        w_phy  = getattr(self, "w_phy",  1.0)
        w_ic   = getattr(self, "w_ic",   1.0)
        w_data = getattr(self, "w_data", 1.0)
        w_seq  = getattr(self, "w_seq",  1.0)
        w_b    = getattr(self, "w_b",    1.0)
        
        total = (
             w_phy * loss_phy_s
             + w_ic * loss_ic_s
             + w_data * data_loss_s
             + w_seq * seq_loss_s
             + w_b * L_smooth_s         
         )
        # + (λ_teacher * L_teacher_s if (λ_teacher is not None) else 0.0)
        #total = data_loss_s



        dbg = {
            "phy":    float(loss_phy_s.detach().item()),
            "ic":     float(loss_ic_s.detach().item()),
            "seq":    float(seq_loss_s.detach().item()),
            "data":   float(data_loss_s.detach().item()),
            "smooth": float(L_smooth_s.detach().item()),
            "teacher": float(L_teacher_s.detach().item()),
            "teacher_stride": int(teacher_stride) if teacher_stride is not None else 1,
            "pm":     float(_to_scalar(loss_pm).detach().item()),
            "ra":     float(_to_scalar(loss_ra).detach().item()),
            "rb":     float(_to_scalar(loss_rb).detach().item()),
            "cs":     float(_to_scalar(loss_cs).detach().item()),
            "st":     float(_to_scalar(loss_st).detach().item()),
            "end":    float(_to_scalar(loss_end).detach().item()),
            "hou":    float(_to_scalar(loss_hou).detach().item()),
            "coolant_on_mean": float(coolant_on.mean().detach().item()),
            "T_pm_tensor": T_pm,
            "T_cs_tensor": T_cs,
            "T_ra_tensor": T_ra,
            "T_rb_tensor": T_rb,
            "T_st_tensor": T_st,
            "T_end_tensor": T_end,
            "T_hou_tensor": T_hou,
            "t_np_used": t_np,
            "N_used": int(N),
        }

        if getattr(self, "debug_seq", False):
            print("[DBG] N_used:", dbg["N_used"], "t0,tend:", dbg["t_np_used"][0], dbg["t_np_used"][-1])
            print("[DBG] phy=", dbg["phy"], "ic=", dbg["ic"], "seq=", dbg["seq"], "data=", dbg["data"], "teacher=", dbg["teacher"], "smooth=", dbg["smooth"])
            print("[DBG] λ_seq=", float(λ_seq), "λ_data=", float(λ_data), "λ_teacher=", float(λ_teacher), "stride=", dbg["teacher_stride"])

        return total, dbg
    




    def predict(self, loss_sequence, time_axis, domain_id=None, enforce_ic: bool = True):
        import numpy as np
        import torch
        self.eval()

        model_dtype = next(self.parameters()).dtype
        device = self.device

        # --- 1) loss_sequence tensor ---
        if not torch.is_tensor(loss_sequence):
            loss_sequence = torch.as_tensor(loss_sequence, dtype=model_dtype, device=device)
        else:
            loss_sequence = loss_sequence.to(device=device, dtype=model_dtype)

        if loss_sequence.dim() != 3 or loss_sequence.size(0) != 1:
            raise ValueError(f"loss_sequence must be [1,T,D], got {tuple(loss_sequence.shape)}")

        # --- 2) time_axis -> t_phys, t_norm, idx ---
        t_phys = torch.as_tensor(time_axis, dtype=model_dtype, device=device).view(-1)  # [N]
        N = int(t_phys.numel())
        if N == 0:
            return np.array([]), np.array([])

        t_norm = (t_phys / PREDICT_TIME).clamp(0.0, 1.0).view(-1, 1)  # [N,1]

        T_all = int(loss_sequence.size(1))
        if T_all < 2:
            idx = torch.zeros((N,), dtype=torch.long, device=device)
        else:
            dt_base = PREDICT_TIME / float(T_all - 1)
            idx = torch.round((t_phys / dt_base)).to(torch.long)
            idx = torch.clamp(idx, 0, T_all - 1)

        # --- 3) ✅ GRU는 "전체 시퀀스"에 먼저 돌린다 ---
        if domain_id is None:
            domain_id = torch.zeros(1, dtype=torch.long, device=device)
        elif not torch.is_tensor(domain_id):
            domain_id = torch.as_tensor(domain_id, dtype=torch.long, device=device).view(-1)
        else:
            domain_id = domain_id.to(device=device, dtype=torch.long).view(-1)
        if domain_id.numel() != 1:
            domain_id = domain_id[:1]

        h_full = self.gru_encoder(loss_sequence, domain_id=domain_id, return_sequences=True).squeeze(0)  # [T_all, H]

        # --- 4) ✅ 그 다음에 시간축으로 index 해서 BN 입력 구성 ---
        h_seq = h_full.index_select(0, idx)  # [N, H]
        bout = self.BN(h_seq)               # [N, C]

        # (옵션) branch smoothing
        if getattr(self, "smooth_branch", False) and N >= 2:
            b = bout
            bout = torch.cat([b[:1], 0.5 * b[1:] + 0.5 * b[:-1]], dim=0)

        # --- 5) trunk ---
        phi_pm = self.PM(t_norm).to(model_dtype)
        # coil-hot은 CS로 통일 (END 금지)
        phi_hot = self.CS(t_norm).to(model_dtype)



        # --- 6) output ---
        dTpm = self.output(bout, phi_pm).squeeze(1) * T_MAX
        dTh  = self.output(bout, phi_hot).squeeze(1) * T_MAX

        Tamb_vec = torch.full((N,), 20.0, dtype=model_dtype, device=device)
        Tpm  = (dTpm + Tamb_vec).detach().cpu().numpy()
        Thot = (dTh  + Tamb_vec).detach().cpu().numpy()

        if enforce_ic and N > 0:
            Tpm[0] = 20.0
            Thot[0] = 20.0

        return Tpm, Thot










    def predict_rollout(self, loss_sequence, time_axis, domain_id=None, enforce_ic: bool = True,
                        dt_internal: float = 1.0,  # ✅ 내부 적분 time step (1s 권장, 속도 필요하면 10s)
                        rollout_end_time: float = None,  # ✅ None이면 max(time_axis) 사용, 아니면 강제로 끝시간(예: 3600)
                        return_full: bool = False):  # ✅ True면 dense 전체를 반환(디버그/플롯용)
        """
        Rollout (Euler forward) with dense internal integration even if time_axis is sparse.

        - If input time_axis is sparse (e.g., GT points), we integrate on dense grid:
            t_internal = 0:dt_internal:rollout_end_time (default=max(time_axis))
        and then sample outputs at the original time_axis points for return.

        Return:
        if return_full == False:
            (Tpm_out[N], Tcs_out[N], Tend_out[N]) aligned to input time_axis (N=len(time_axis))
        else:
            (t_internal, Tpm_dense, Tcs_dense, Tend_dense) for full trajectory
        """
        import numpy as np
        import torch

        was_training = self.training
        self.eval()

        # ---- helpers ----
        def _to_scalar_tensor(x, default=0.0, dtype=torch.float64):
            if x is None:
                return torch.tensor(float(default), dtype=dtype, device=self.device)
            if torch.is_tensor(x):
                return x.reshape(-1)[0].to(dtype=dtype, device=self.device)
            return torch.tensor(float(x), dtype=dtype, device=self.device)

        def _clamp_R(R):
            return torch.clamp(R, min=1e-6)

        # -----------------------------
        # 0) sanitize input time_axis
        # -----------------------------
        time_axis = np.asarray(time_axis, dtype=np.float64).reshape(-1)
        time_axis = np.unique(time_axis)  # 중복 제거
        time_axis = np.sort(time_axis)

        if time_axis.size < 2:
            raise ValueError("time_axis must have at least 2 points.")

        if np.any(np.diff(time_axis) <= 0):
            raise ValueError("time_axis must be strictly increasing after sanitize.")

        # ------------------------------------------
        # 1) build dense internal integration timeline
        # ------------------------------------------
        if rollout_end_time is None:
            t_end = float(time_axis[-1])
        else:
            t_end = float(rollout_end_time)

        t0 = float(time_axis[0])
        if t0 != 0.0:
            # 일반적으로 0부터 시작을 기대하므로, 필요 시 0 포함
            t0 = 0.0

        dt_internal = float(dt_internal)
        if dt_internal <= 0:
            raise ValueError("dt_internal must be positive.")

        # internal dense axis
        t_internal = np.arange(t0, t_end + 1e-12, dt_internal, dtype=np.float64)
        if t_internal.size < 2:
            raise ValueError("Internal time grid too small. Check dt_internal / t_end.")

        # mapping: sample dense outputs at requested time_axis
        # (가까운 오른쪽 인덱스 선택)
        sample_idx = np.searchsorted(t_internal, time_axis, side="left")
        sample_idx = np.clip(sample_idx, 0, t_internal.size - 1)

        with torch.no_grad():
            # ===== fixed reference temperatures (paper-aligned) =====
            Tamb0  = 20.0
            Tcool0 = 30.0
            T_amb  = torch.tensor(Tamb0,  dtype=torch.float64, device=self.device)
            T_cool = torch.tensor(Tcool0, dtype=torch.float64, device=self.device)

            # ===== state: ΔT relative to Tamb =====
            dTpm  = torch.tensor(0.0, dtype=torch.float64, device=self.device)
            dTrb  = torch.tensor(0.0, dtype=torch.float64, device=self.device)  # RotorB
            dTra  = torch.tensor(0.0, dtype=torch.float64, device=self.device)  # RotorA (NEW)
            dTcs  = torch.tensor(0.0, dtype=torch.float64, device=self.device)  # CS
            dTst  = torch.tensor(0.0, dtype=torch.float64, device=self.device)  # Stator
            dTend = torch.tensor(0.0, dtype=torch.float64, device=self.device)  # End-coil
            dTh   = torch.tensor(0.0, dtype=torch.float64, device=self.device)  # Housing

            # outputs (absolute temperature) on dense internal grid
            Tpm_dense  = np.empty((t_internal.size,), dtype=np.float64)
            Tcs_dense  = np.empty((t_internal.size,), dtype=np.float64)
            Tend_dense = np.empty((t_internal.size,), dtype=np.float64)

            # initial store
            Tpm_dense[0]  = Tamb0
            Tcs_dense[0]  = Tamb0
            Tend_dense[0] = Tamb0

            # fixed split for copper loss
            V_cs  = 4.899e-4
            V_end = 9.024e-4
            f_cs  = V_cs  / (V_cs + V_end)
            f_end = V_end / (V_cs + V_end)

            # ===== constants / buffers =====
            C_pm  = self.C_pm.to(torch.float64).to(self.device)
            C_rb  = (self.C_rotorB if hasattr(self, "C_rotorB") else self.C_rotor).to(torch.float64).to(self.device)
            C_cs  = self.C_CS.to(torch.float64).to(self.device)
            C_st  = self.C_stator.to(torch.float64).to(self.device)
            C_end = self.C_end.to(torch.float64).to(self.device)

            if hasattr(self, "C_housing"):
                C_hou = self.C_housing.to(torch.float64).to(self.device)
            else:
                C_hou = torch.tensor(28823.0319, dtype=torch.float64, device=self.device)

            # ===== resistances (base) =====
            R_pm_rb  = (self.R_pm_rb if hasattr(self, "R_pm_rb") else self.R_rt_pm).to(torch.float64).to(self.device)
            R_cs_st  = (self.R_cs_st if hasattr(self, "R_cs_st") else self.R_ch_st).to(torch.float64).to(self.device)
            R_cs_end = self.R_cs_end.to(torch.float64).to(self.device)

            R_pm_rb  = _clamp_R(R_pm_rb)
            R_cs_st  = _clamp_R(R_cs_st)
            R_cs_end = _clamp_R(R_cs_end)

            # ===== End/PM -> Air convection =====
            h_air = (self.h_air if hasattr(self, "h_air") else torch.tensor(20.0)).to(torch.float64).to(self.device)
            A_end_surf = (self.A_end_surf if hasattr(self, "A_end_surf") else torch.tensor(0.01023)).to(torch.float64).to(self.device)
            A_pm_surf  = (self.A_pm_surf  if hasattr(self, "A_pm_surf")  else torch.tensor(0.00156)).to(torch.float64).to(self.device)

            R_end_amb = 1.0 / (h_air * A_end_surf + 1e-12)
            R_pm_amb  = 1.0 / (h_air * A_pm_surf  + 1e-12)
            R_end_amb = _clamp_R(R_end_amb)
            R_pm_amb  = _clamp_R(R_pm_amb)

            # ===== Air-gap params for R_rb_cs =====
            k_air      = self.k_air.to(torch.float64).to(self.device)
            delta_gap  = self.delta_gap.to(torch.float64).to(self.device)
            A_rb_cs    = self.A_rb_cs.to(torch.float64).to(self.device)
            h_conv0    = self.h_conv0.to(torch.float64).to(self.device)
            h_conv_exp = self.h_conv_exp.to(torch.float64).to(self.device)
            rpm_ref    = self.RPM_NORM_REF.to(torch.float64).to(self.device)

            # ===== RotorA params =====
            C_ra = (self.C_rotorA if hasattr(self, "C_rotorA") else 0.3 * C_rb).to(torch.float64).to(self.device)
            C_ra = torch.clamp(C_ra, min=1e-6)
            R_pm_rb2 = (self.R_pm_rb if hasattr(self, "R_pm_rb") else torch.tensor(0.85)).to(torch.float64).to(self.device)
            R_ra_pm  = (self.R_ra_pm if hasattr(self, "R_ra_pm") else torch.tensor(0.85)).to(torch.float64).to(self.device)
            R_pm_rb2 = _clamp_R(R_pm_rb2)
            R_ra_pm  = _clamp_R(R_ra_pm)

            # ===== Housing thermal resistances =====
            R_st_hou  = (self.R_st_hou  if hasattr(self, "R_st_hou")  else torch.tensor(2.0)).to(torch.float64).to(self.device)
            R_end_hou = (self.R_end_hou if hasattr(self, "R_end_hou") else torch.tensor(0.126)).to(torch.float64).to(self.device)
            R_hou_amb = (self.R_hou_amb if hasattr(self, "R_hou_amb") else torch.tensor(1000.0)).to(torch.float64).to(self.device)
            R_st_hou  = _clamp_R(R_st_hou)
            R_end_hou = _clamp_R(R_end_hou)
            R_hou_amb = _clamp_R(R_hou_amb)

            # time loop on dense grid
            for i in range(t_internal.size - 1):
                t_curr = float(t_internal[i])
                dt = float(t_internal[i + 1] - t_internal[i])
                # dt is fixed = dt_internal (except last numerical issues)
                dt_t = torch.tensor(dt, dtype=torch.float64, device=self.device)

                # normalized time for interpolation function
                t_norm = torch.tensor([t_curr / PREDICT_TIME], dtype=torch.float64, device=self.device)

                # --- get q/h/LPM/RPM ---
                q_mag, q_rot, q_cop, q_sta, h_t, LPM_t, Tamb_t, RPM_t = self._interp_q_at(t_norm)

                q_mag_s = _to_scalar_tensor(q_mag, 0.0)
                q_rot_s = _to_scalar_tensor(q_rot, 0.0)
                q_cop_s = _to_scalar_tensor(q_cop, 0.0)
                q_sta_s = _to_scalar_tensor(q_sta, 0.0)

                LPM_s = _to_scalar_tensor(LPM_t, 0.0)
                RPM_s = _to_scalar_tensor(RPM_t, 0.0)

                if i == 0:
                    print("[DEBUG Q]",
                        "q_cop =", float(q_cop_s.item()),
                        "q_mag =", float(q_mag_s.item()),
                        "q_sta =", float(q_sta_s.item()),
                        "LPM   =", float(LPM_s.item()),
                        "RPM   =", float(RPM_s.item()))

                # water cooling OFF if LPM=0
                LPM_EPS = 1e-6
                is_flow_off = (float(LPM_s.item()) <= LPM_EPS)

                # NOTE: 기존 코드대로 유지 (원하면 eff 값 조정 가능)
                R_end_amb_eff = torch.tensor(1e12, dtype=torch.float64, device=self.device) if is_flow_off else R_end_amb

                # --- dynamic water h (stator cooling) ---
                h_dyn = None
                if getattr(self, "use_dynamic_h", False):
                    if not is_flow_off:
                        if h_t is not None:
                            h_dyn = _to_scalar_tensor(h_t, 0.0)
                        else:
                            h_dyn = self._h_from_LPM_tcool_rpm_dual(
                                LPM_s.view(1),
                                torch.tensor([Tcool0], dtype=torch.float64, device=self.device),
                                RPM_s.view(1)
                            ).view(1)[0]

                    else:
                        h_dyn = None

                # --- Stator->coolant resistance: R = R_st_ch + 1/(h*A) ---
                R_st_cool = None
                if h_dyn is not None:
                    h_dyn = torch.clamp(h_dyn, min=1e-6)

                    if hasattr(self, "A_st_ch"):
                        A_st_use = self.A_st_ch.to(torch.float64).to(self.device)
                    elif hasattr(self, "A_st"):
                        A_st_use = self.A_st.to(torch.float64).to(self.device)
                    else:
                        A_st_use = None

                    if A_st_use is not None:
                        R_conv = 1.0 / (h_dyn * A_st_use + 1e-12)
                        if hasattr(self, "R_st_ch"):
                            R_wall = self.R_st_ch.to(torch.float64).to(self.device)
                            R_st_cool = _clamp_R(R_wall + R_conv)
                        else:
                            R_st_cool = _clamp_R(R_conv)

                # --- Air-gap RB<->CS resistance (dynamic) ---
                h_cond = k_air / (delta_gap + 1e-12)
                rpm_n = torch.clamp(RPM_s / (rpm_ref + 1e-12), min=0.0)
                h_conv_rot = h_conv0 * (rpm_n ** h_conv_exp)
                h_gap_eff = h_cond + h_conv_rot
                R_rb_cs = 1.0 / (h_gap_eff * A_rb_cs + 1e-12)
                R_rb_cs = _clamp_R(R_rb_cs)

                # --- copper loss split ---
                q_cop_cs  = q_cop_s * f_cs
                q_cop_end = q_cop_s * f_end

                # --- absolute temperatures ---
                Tpm  = T_amb + dTpm
                Trb  = T_amb + dTrb
                Tra  = T_amb + dTra
                Tcs  = T_amb + dTcs
                Tst  = T_amb + dTst
                Tend = T_amb + dTend
                Th   = T_amb + dTh

                # ===== RHS =====
                dTpm_dt = ( q_mag_s
                            - (Tpm - Trb) / R_pm_rb
                            - (Tpm - Tra) / R_ra_pm
                            - (Tpm - T_amb) / R_pm_amb
                        ) / C_pm

                dTrb_dt = ( q_rot_s
                            - (Trb - Tpm) / R_pm_rb
                            - (Trb - Tcs) / R_rb_cs
                        ) / C_rb

                dTra_dt = ( q_rot_s
                            - (Tra - Tpm) / R_ra_pm
                        ) / C_ra

                dTcs_dt = ( q_cop_cs
                            - (Tcs - Trb) / R_rb_cs
                            - (Tcs - Tst) / R_cs_st
                            - (Tcs - Tend) / R_cs_end
                        ) / C_cs

                base_st = ( q_sta_s
                            - (Tst - Tcs) / R_cs_st
                            - (Tst - Th)  / R_st_hou )

                if R_st_cool is not None:
                    dTst_dt = ( base_st - (Tst - T_cool) / R_st_cool ) / C_st
                else:
                    dTst_dt = base_st / C_st

                dTend_dt = ( q_cop_end
                            - (Tend - Tcs)   / R_cs_end
                            - (Tend - T_amb) / R_end_amb_eff
                            - (Tend - Th)    / R_end_hou
                        ) / C_end

                dTh_dt = ( (Tst - Th)   / R_st_hou
                        + (Tend - Th) / R_end_hou
                        - (Th - T_amb)/ R_hou_amb ) / C_hou

                # ===== integrate ΔT =====
                dTpm  = dTpm  + dt_t * dTpm_dt
                dTrb  = dTrb  + dt_t * dTrb_dt
                dTra  = dTra  + dt_t * dTra_dt
                dTcs  = dTcs  + dt_t * dTcs_dt
                dTst  = dTst  + dt_t * dTst_dt
                dTend = dTend + dt_t * dTend_dt
                dTh   = dTh   + dt_t * dTh_dt

                # store dense outputs at next index (i+1)
                Tpm_dense[i + 1]  = float(Tamb0 + dTpm.item())
                Tcs_dense[i + 1]  = float(Tamb0 + dTcs.item())
                Tend_dense[i + 1] = float(Tamb0 + dTend.item())

            # enforce IC on dense
            if enforce_ic and Tpm_dense.size > 0:
                Tpm_dense[0]  = Tamb0
                Tcs_dense[0]  = Tamb0
                Tend_dense[0] = Tamb0

        # ------------------------------------------
        # 2) return
        # ------------------------------------------
        if return_full:
            if was_training:
                self.train()
            return t_internal, Tpm_dense, Tcs_dense, Tend_dense

        # sample at requested input time_axis points
        Tpm_out  = Tpm_dense[sample_idx]
        Tcs_out  = Tcs_dense[sample_idx]
        Tend_out = Tend_dense[sample_idx]

        print("time_axis(in) shape:", time_axis.shape, "internal:", t_internal.shape)
        print("Tcs_roll(out) shape:", Tcs_out.shape, "sample:", Tcs_out[:min(5, Tcs_out.size)])

        if was_training:
            self.train()

        return Tpm_out, Tcs_out, Tend_out







def min_max_scale(data, min_val, max_val, eps=1e-8):
    return (data - min_val) / (max_val - min_val + eps)


def denorm_channel(q_norm, key):
    qmin, qmax = LOSS_MIN_MAX[key]
    return q_norm * (qmax - qmin) + qmin

def load_gt_with_time(gt_path: str) -> pd.DataFrame:
    # 1) 1차 로드 (기본 sep=',' 가정)
    df = pd.read_csv(gt_path)

    # 2) 컬럼명 정리 (BOM/공백 제거)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)  # BOM 제거
        .str.strip()                              # 공백 제거
    )

    # 3) sep 문제 감지: 컬럼이 1개인데, 그 안에 ; 또는 \t 가 있으면 재로딩
    if "time" not in df.columns and len(df.columns) == 1:
        header = df.columns[0]
        if ";" in header:
            df = pd.read_csv(gt_path, sep=";")
        elif "\t" in header:
            df = pd.read_csv(gt_path, sep="\t")

        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
        )

    # 4) 흔한 대체명 매핑 (Time/t/sec/seconds 등)
    if "time" not in df.columns:
        lower_map = {c.lower().strip(): c for c in df.columns}
        for alt in ["time", "t", "sec", "secs", "second", "seconds"]:
            if alt in lower_map:
                df = df.rename(columns={lower_map[alt]: "time"})
                break

    # 5) time이 인덱스로 들어온 케이스 처리
    if "time" not in df.columns:
        if df.index.name and df.index.name.lower().strip() in ["time", "t", "sec", "secs", "seconds"]:
            df = df.reset_index()
            df.columns = ["time"] + list(df.columns[1:])
        elif "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "time"})

    # 6) 최종 체크
    if "time" not in df.columns:
        raise KeyError(f"[GT] No 'time' column in {gt_path}. columns={list(df.columns)}")

    # 7) time numeric 보장
    df["time"] = pd.to_numeric(df["time"], errors="coerce")

    return df



def load_and_prepare_sequence(loss_csv_path, t_amb_fill=None):
    """
    CSV → 1초 보간 → 7채널 시퀀스 생성
    채널: [mag, rotor, stator, copper, LPM, T_amb, RPM]

    - Tamb는 항상 20°C 고정 (CSV Tamb 컬럼이 있어도 무시)
    - CSV에 LPM, rpm_v(RPM) 열이 없으면 0으로 채움
    반환: torch.Tensor [1, 3601, 7]
    """
    df = pd.read_csv(loss_csv_path)
    time_grid = np.arange(0, PREDICT_TIME + 1e-9, 1.0)

    required = ["time", "magnet_loss", "rotor_loss", "copper_loss", "stator_loss"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {loss_csv_path}. Available: {list(df.columns)}")

    t_src = df["time"].values

    # 4개 손실 보간
    q_mag_raw = np.interp(time_grid, t_src, df["magnet_loss"].values)
    q_rot_raw = np.interp(time_grid, t_src, df["rotor_loss"].values)
    q_cop_raw = np.interp(time_grid, t_src, df["copper_loss"].values)
    q_sta_raw = np.interp(time_grid, t_src, df["stator_loss"].values)

    # LPM 보간
    if "LPM" in df.columns:
        LPM_raw = np.interp(time_grid, t_src, df["LPM"].values)
    else:
        LPM_raw = np.full_like(time_grid, 0.0, dtype=np.float64)

    # Tamb: 항상 20 고정 (정규화 키가 없어도 안전하게 처리)
    Tamb_raw = np.full_like(time_grid, 20.0, dtype=np.float64)

    # RPM 보간
    if "rpm_v" in df.columns:
        RPM_raw = np.interp(time_grid, t_src, df["rpm_v"].values)
    elif "RPM" in df.columns:
        RPM_raw = np.interp(time_grid, t_src, df["RPM"].values)
    else:
        RPM_raw = np.full_like(time_grid, 0.0, dtype=np.float64)

    # 정규화 (필수 키들)
    q_mag = min_max_scale(q_mag_raw, LOSS_MIN_MAX["mag"][0], LOSS_MIN_MAX["mag"][1])
    q_rot = min_max_scale(q_rot_raw, LOSS_MIN_MAX["rot"][0], LOSS_MIN_MAX["rot"][1])
    q_sta = min_max_scale(q_sta_raw, LOSS_MIN_MAX["sta"][0], LOSS_MIN_MAX["sta"][1])
    q_cop = min_max_scale(q_cop_raw, LOSS_MIN_MAX["cop"][0], LOSS_MIN_MAX["cop"][1])

    # LPM 키가 없을 수도 있으니 방어
    if "LPM" in LOSS_MIN_MAX:
        LPM_n = min_max_scale(LPM_raw, LOSS_MIN_MAX["LPM"][0], LOSS_MIN_MAX["LPM"][1])
    else:
        LPM_n = LPM_raw.astype(np.float64)

    # Tamb: 고정 20이므로 굳이 min-max 안 해도 됨 (키 없어도 OK)
    # - 가장 안전한 선택: 20 그대로 넣기
    Tamb_n = Tamb_raw.astype(np.float64)

    # RPM: 키 있으면 정규화, 없으면 raw 그대로
    if "RPM" in LOSS_MIN_MAX:
        RPM_n = min_max_scale(RPM_raw, LOSS_MIN_MAX["RPM"][0], LOSS_MIN_MAX["RPM"][1])
    else:
        RPM_n = RPM_raw.astype(np.float64)

    # [T, 7] → [1, T, 7]
    seq7 = np.stack([q_mag, q_rot, q_sta, q_cop, LPM_n, Tamb_n, RPM_n], axis=1).astype(np.float64)
    return torch.from_numpy(seq7).unsqueeze(0).to(DEVICE)



# ===== legacy single-GRU -> dual-GRU ckpt migration loader =====
def load_legacy_single_gru_ckpt(model, ckpt_path, map_location=None):
    """
    레거시(싱글 GRU) 체크포인트를 듀얼 GRU 모델로 로드:
      - gru_encoder.gru.*  → gru_encoder.gru_var.* / gru_encoder.gru_fix.* 로 복제
      - h0_var / h0_fix 은 0으로 초기화(모델 파라미터 기본값 사용)
      - 나머지 키는 동일하면 그대로 복사
    """
    import torch
    sd_old = torch.load(ckpt_path, map_location=map_location)

    # 현재 모델의 키/shape를 기준으로 새 dict 생성
    sd_new = model.state_dict()  # (복사본 아님 주의) → 업데이트 후 load_state_dict에 넘길 용도

    # 1) GRU 외의 키들은 이름이 그대로면 덮어쓰기
    for k, v in sd_old.items():
        if not k.startswith("gru_encoder.gru."):
            if k in sd_new and sd_new[k].shape == v.shape:
                sd_new[k] = v  # shape 맞으면 교체

    # 2) GRU 가중치 매핑: gru -> (gru_var, gru_fix) 로 복제
    #    레이어 수/양식은 현재 모델에서 읽어온다.
    try:
        n_layers = model.gru_encoder.gru_var.num_layers
    except AttributeError:
        # 혹시 듀얼 GRU가 아닌 경우 보호
        raise RuntimeError("현재 모델이 듀얼 GRU 구조가 아닙니다( gru_var / gru_fix 확인 ).")

    suffixes = []
    # 각 레이어별 weight/bias 4종
    base_sufs = ["weight_ih_l{L}", "weight_hh_l{L}", "bias_ih_l{L}", "bias_hh_l{L}"]
    for L in range(n_layers):
        for pat in base_sufs:
            suffixes.append(pat.format(L=L))

    for suf in suffixes:
        old_key = f"gru_encoder.gru.{suf}"     # 레거시 키
        if old_key in sd_old:
            w = sd_old[old_key]
            # 두 도메인에 동일 가중치로 초기화
            sd_new[f"gru_encoder.gru_var.{suf}"] = w
            sd_new[f"gru_encoder.gru_fix.{suf}"] = w
        # old_key 없으면(예: 레이어 수 달라짐) 그냥 넘어가고 모델 초기값 사용

    # 3) h0_var / h0_fix 는 모델 초기 파라미터(0) 그대로 둔다.
    #    (이미 sd_new 안에 존재하므로 건드릴 필요 없음)

    # 4) 로드 (엄격 불가: 일부 키 스킵/추가 있을 수 있음)
    missing, unexpected = model.load_state_dict(sd_new, strict=False)
    # 원한다면 로깅:
    if missing or unexpected:
        print("[load_legacy_single_gru_ckpt] missing:", missing)
        print("[load_legacy_single_gru_ckpt] unexpected:", unexpected)





def _infer_domain_id(name: str) -> int:
    name_lower = name.lower()
    # 예: "variable1(train)" / "fixed4(test)"
    return 0 if "variable" in name_lower else 1  # 0=variable, 1=fixed

if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    EXP_NAME = "exp4_noLbLs"

    save_dir = os.path.join(SAVE_DIR_BASE, script_name + "_" + EXP_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # ===== Tamb 고정 (실험 설정) =====
    T_AMB_FIXED = 20.0

    datasets_info = [
        {"name": "Case1_0(train)", "loss_csv": os.path.join(DATA_DIR, "Case1_0.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case1_0.csv")},
        {"name": "Case1_1(train)", "loss_csv": os.path.join(DATA_DIR, "Case1_1.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case1_1.0.csv")},
        {"name": "Case3_0(train)", "loss_csv": os.path.join(DATA_DIR, "Case3_0.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case3_0.csv")},
        {"name": "Case2_0(train)", "loss_csv": os.path.join(DATA_DIR, "Case2_0.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case2_0.csv")},
    ]

    # test_dataset_info = [
    #     {"name": "variable(test2)", "loss_csv": os.path.join(DATA_DIR, "predicted_loss2.csv"),
    #      "gt_merged": os.path.join(DATA_DIR, "ground_truth2.csv")},
    #     {"name": "variable(test4)", "loss_csv": os.path.join(DATA_DIR, "predicted_loss4.csv"),
    #      "gt_merged": os.path.join(DATA_DIR, "ground_truth4.csv")},
    #     {"name": "fixed1(test)", "loss_csv": os.path.join(DATA_DIR, "predicted_loss_fixed4_4.csv"),
    #      "gt_merged": os.path.join(DATA_DIR, "data_only_avg_1.csv")},
    #     {"name": "fixed3(test)", "loss_csv": os.path.join(DATA_DIR, "predicted_loss_fixed6.csv"),
    #      "gt_merged": os.path.join(DATA_DIR, "data_only_avg_3.csv")},
    #     {"name": "fixed4(test)", "loss_csv": os.path.join(DATA_DIR, "predicted_loss_fixed7.csv"),
    #      "gt_merged": os.path.join(DATA_DIR, "data_only_avg_4.csv")},
    #     {"name": "fixed5(test)", "loss_csv": os.path.join(DATA_DIR, "predicted_loss_fixed8.csv"),
    #      "gt_merged": os.path.join(DATA_DIR, "data_only_avg_5.csv")},
    # ]


    test_dataset_info = [
        {"name": "Case1_0.2(test)", "loss_csv": os.path.join(DATA_DIR, "Case1_0.2.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case1_0.2.csv")},
        {"name": "Case1_0.4(test)", "loss_csv": os.path.join(DATA_DIR, "Case1_0.4.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case1_0.4.csv")},
        {"name": "Case1_0.6(test)", "loss_csv": os.path.join(DATA_DIR, "Case1_0.6.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case1_0.6.csv")},
        {"name": "Case1_0.8(test)", "loss_csv": os.path.join(DATA_DIR, "Case1_0.8.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case1_0.8.csv")},
        {"name": "Case2_0.2(test)", "loss_csv": os.path.join(DATA_DIR, "Case2_0.2.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case2_0.2.csv")},
        {"name": "Case2_0.4(test)", "loss_csv": os.path.join(DATA_DIR, "Case2_0.4.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case2_0.4.csv")},
        {"name": "Case2_0.6(test)", "loss_csv": os.path.join(DATA_DIR, "Case2_0.6.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case2_0.6.csv")},
        {"name": "Case2_0.8(test)", "loss_csv": os.path.join(DATA_DIR, "Case2_0.8.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case2_0.8.csv")},
        {"name": "Case2_1(test)", "loss_csv": os.path.join(DATA_DIR, "Case2_1.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case2_1.0.csv")},
        {"name": "Case3_0.2(test)", "loss_csv": os.path.join(DATA_DIR, "Case3_0.2.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case3_0.2.csv")},
        {"name": "Case3_0.4(test)", "loss_csv": os.path.join(DATA_DIR, "Case3_0.4.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case3_0.4.csv")},
        {"name": "Case3_0.6(test)", "loss_csv": os.path.join(DATA_DIR, "Case3_0.6.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case3_0.6.csv")},
        {"name": "Case3_0.8(test)", "loss_csv": os.path.join(DATA_DIR, "Case3_0.8.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case3_0.8.csv")},
        {"name": "Case3_1(test)", "loss_csv": os.path.join(DATA_DIR, "Case3_1.csv"),
         "gt_merged": os.path.join(DATA_DIR2, "gt_Case3_1.0.csv")},



    ]

    

    all_datasets_info = datasets_info + test_dataset_info
    # all_datasets_info = datasets_info 


    print("모든 데이터셋을 메모리로 미리 로딩합니다...")

    # --- helper: variable/fixed → domain_id(0/1) 추론 ---
    def _infer_domain_id(info: dict) -> int:
        dom = (info.get("domain") or "").lower()
        if dom.startswith("var"): return 0
        if dom.startswith("fix"): return 1
        s = f"{info.get('name','')} {info.get('loss_csv','')} {info.get('gt_merged','')}".lower()
        if "fixed" in s or "fix" in s: return 1
        if "variable" in s or "var" in s: return 0
        return 0

    # --- preload block ---
    all_data_preloaded = []
    for ds_info in all_datasets_info:
        # ★ Tamb는 20 고정: load_and_prepare_sequence 내부도 20으로 고정되어야 함
        loss_seq_tensor = load_and_prepare_sequence(ds_info["loss_csv"], t_amb_fill=T_AMB_FIXED)
        # 기대: (1, 3601, 7)
        assert loss_seq_tensor.dim() == 3 and loss_seq_tensor.size(-1) == 7, \
            f"loss_sequence must be [1,T,7], got {tuple(loss_seq_tensor.shape)}"
        

        df_gt = load_gt_with_time(ds_info["gt_merged"])
        df_gt = df_gt[df_gt["time"] <= PREDICT_TIME].copy()

        if not (df_gt["time"].values == 0).any():
            # pm_tmax / coil_tmax 컬럼이 있다고 가정
            df0 = pd.DataFrame([{
                "time": 0.0,
                "pm_tmax": float(T_AMB_FIXED),     # 20.0
                "coil_tmax": float(T_AMB_FIXED),   # 20.0
            }])
            df_gt = pd.concat([df0, df_gt], ignore_index=True).sort_values("time").reset_index(drop=True)



        print("[DBG] df_gt.shape:", df_gt.shape)
        print("[DBG] df_gt.columns:", list(df_gt.columns))
        print("[DBG] df_gt.head():\n", df_gt.head(3))


        dom_id = _infer_domain_id(ds_info)

        ds_info = dict(ds_info)
        ds_info["domain_id"] = dom_id
        ds_info["t_amb"] = T_AMB_FIXED  # ★ 강제

        all_data_preloaded.append({
            "info": ds_info,
            "loss_sequence": loss_seq_tensor,     # (1, 3601, 7)
            "df_gt": df_gt,
            "t_train": df_gt["time"].values,
            "domain_id": dom_id,
        })

    train_data_preloaded = [d for d in all_data_preloaded if "train" in d["info"]["name"]]
    test_data_preloaded  = [d for d in all_data_preloaded if "test"  in d["info"]["name"]]
    assert len(train_data_preloaded) > 0, "학습 데이터가 없습니다."

    # ===== 모델 생성 =====
    model = HybridModel()
    model.eval()
    # 실험 1: 전체 (논문 그대로 5개 다)
    #model.w_phy, model.w_ic, model.w_data, model.w_seq, model.w_b = 1, 1, 1, 1, 1

    # 실험 2: Ls(seq) 제외
    #model.w_phy, model.w_ic, model.w_data, model.w_seq, model.w_b = 1, 1, 1, 0, 1

    # 실험 3: Lb(smooth) 제외
    #model.w_phy, model.w_ic, model.w_data, model.w_seq, model.w_b = 1, 1, 1, 1, 0

    # 실험 4: Ls, Lb 동시 제외
    model.w_phy, model.w_ic, model.w_data, model.w_seq, model.w_b = 1, 1, 1, 0, 0
    
    print(f"Device: {DEVICE}")
    print("====== 1단계: Adam을 사용한 기본 학습 시작 ======")

    # 커리큘럼 스위치 하이퍼
    PHYS_THRESH   = 1e9
    EMA_MOMENTUM  = 0.98
    W_DATA_MAX    = 1.0
    RAMP_EPOCHS   = 3000  
    GRAD_CLIP     = 1.0

    ema_phy        = 0.0
    use_data       = True
    epoch_switched = 0

    def _ramp_weight(iter_idx, start_iter, ramp_iters, w_max):
        if (start_iter is None) or (iter_idx < start_iter):
            return 0.0
        r = min(1.0, float(iter_idx - start_iter) / max(1, ramp_iters))
        return float(w_max * r)
    

    # =========================
    # Loss history buffers
    # =========================
    # ===== loss history (phys -> ic -> data) =====
    loss_plot_dir = os.path.join(save_dir, "loss_plots")
    os.makedirs(loss_plot_dir, exist_ok=True)

    loss_hist = {
        "iter": [],
        "phys": [],
        "data": [],
    }

    # ===== Adam 학습 =====
    for it in range(MAX_ITERS):
        model.train()

        # data ramp weight (data phase ON 이후 점진 증가)
        lambda_data_this = _ramp_weight(it, epoch_switched, RAMP_EPOCHS, W_DATA_MAX) if use_data else 0.0

        # ---- zero_grad (grouped optimizers) ----
        for opt in model.optimizers:
            opt.zero_grad(set_to_none=True)

        total_loss_val = 0.0
        per_ds_logs = []

        # ---- dataset loop: accumulate grads ----
        for data in train_data_preloaded:
            model.set_T_amb(T_AMB_FIXED)  # ★ 항상 20
            dom_id_t = torch.tensor([data["domain_id"]], dtype=torch.long, device=DEVICE)

            # (3601,7) normalized channels: [mag, rot, sta, cop, LPM, Tamb, RPM]
            seq = data["loss_sequence"].squeeze(0).detach().cpu().numpy()
            time_grid = np.arange(0, PREDICT_TIME + 1e-9, 1.0)

            q_mag = denorm_channel(seq[:, 0], "mag")
            q_rot = denorm_channel(seq[:, 1], "rot")
            q_sta = denorm_channel(seq[:, 2], "sta")
            q_cop = denorm_channel(seq[:, 3], "cop")
            LPM   = denorm_channel(seq[:, 4], "LPM")
            RPM   = denorm_channel(seq[:, 6], "RPM") if seq.shape[1] >= 7 else np.zeros_like(time_grid)

            # Tamb는 고정 20
            Tamb  = np.full_like(time_grid, T_AMB_FIXED, dtype=np.float64)

            model.set_q_profile(
                time_grid,
                q_mag, q_rot, q_cop, q_sta,
                LPM=LPM,
                RPM=RPM
            )

            # ----- sparse GT -----
            df_curr      = data["df_gt"]
            t_gt_sparse  = df_curr["time"].values
            pm_gt_sparse = df_curr["pm_tmax"].values
            coil_gt_sparse = df_curr["coil_tmax"].values

            # ----- loss -----
            loss_ds, dbg_ds = model.sequential_consistency_loss(
                data["loss_sequence"],
                time_s_np=time_grid,         # 전체 시간축 (0~3600, 1s)
                gt_time_s_np=t_gt_sparse,    # sparse GT 시간축
                Tpm_gt=pm_gt_sparse,
                Tch_gt=coil_gt_sparse,

                domain_id=dom_id_t,

                # 가중치
                λ_ic=1.0,
                λ_seq=1.0,

                # ★ 핵심: data loss는 ramp로 반영
                λ_data=float(lambda_data_this),

                # rollout teacher (원하면 유지)
                λ_teacher=2.0,
                teacher_stride=10
            )

            # gradient accumulation: dataset 평균으로 스케일
            (loss_ds / len(train_data_preloaded)).backward()

            total_loss_val += float(loss_ds.item())

            per_ds_logs.append({
                "name":  data["info"]["name"],
                "total": float(loss_ds.item()),

                # --- node-wise residual (lpnt_single2 topology 기준) ---
                "pm":  float(dbg_ds.get("pm",  0.0)),
                "ra":  float(dbg_ds.get("ra",  0.0)),
                "rb":  float(dbg_ds.get("rb",  0.0)),
                "cs":  float(dbg_ds.get("cs",  0.0)),
                "st":  float(dbg_ds.get("st",  0.0)),
                "end": float(dbg_ds.get("end", 0.0)),
                "hou": float(dbg_ds.get("hou", 0.0)),

                # --- loss components ---
                "phy":  float(dbg_ds.get("phy",  0.0)),
                "ic":   float(dbg_ds.get("ic",   0.0)),
                "seq":  float(dbg_ds.get("seq",  0.0)),
                "data": float(dbg_ds.get("data", 0.0)),
            })

        # ---- group-wise gradient clipping (중요) ----
        # (전체 model.parameters()로 한 번에 clip하면 alpha가 묻힐 수 있어 그룹별 권장)
        torch.nn.utils.clip_grad_norm_(model.gru_encoder.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(model.BN.parameters(), GRAD_CLIP)

        torch.nn.utils.clip_grad_norm_(model.PM.parameters(), GRAD_CLIP)

        if hasattr(model, "ROTOR_A"):
            torch.nn.utils.clip_grad_norm_(model.ROTOR_A.parameters(), GRAD_CLIP)
        if hasattr(model, "ROTOR_B"):
            torch.nn.utils.clip_grad_norm_(model.ROTOR_B.parameters(), GRAD_CLIP)

        torch.nn.utils.clip_grad_norm_(model.CS.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(model.END.parameters(), GRAD_CLIP)

        if hasattr(model, "STATOR"):
            torch.nn.utils.clip_grad_norm_(model.STATOR.parameters(), GRAD_CLIP)

        # alpha는 별도로(좀 더 널널하게) clip
        torch.nn.utils.clip_grad_norm_([model.log_alpha_h, model.log_alpha_rpm], GRAD_CLIP * 5.0)

        # ---- ema phys tracking ----
        phys_avg = sum(rec["phy"] + rec["seq"] + rec["ic"] for rec in per_ds_logs) / max(1, len(per_ds_logs))
        ema_phy  = EMA_MOMENTUM * ema_phy + (1.0 - EMA_MOMENTUM) * float(phys_avg)

        # ---- record avg losses per iteration ----
        avg_phy  = sum(rec["phy"]  for rec in per_ds_logs) / max(1, len(per_ds_logs))
        avg_data = sum(rec["data"] for rec in per_ds_logs) / max(1, len(per_ds_logs))

        loss_hist["iter"].append(int(it))
        loss_hist["phys"].append(float(avg_phy))
        loss_hist["data"].append(float(avg_data))

        # ---- switch on data phase (phys 충분히 내려가면) ----
        if (not use_data) and (ema_phy <= PHYS_THRESH):
            use_data = True
            epoch_switched = it
            print(f"[SWITCH ON] it={it} ema_phy={ema_phy:.3e} ≤ {PHYS_THRESH:.3e} → enable data loss ramp")

        # ---- optimizer step (grouped) ----
        for opt in model.optimizers:
            opt.step()

        # ---- scheduler step (seq only) ----
        model.scheduler_seq.step()

        # ---- alpha safety clamp (매 iter 권장) ----
        with torch.no_grad():
            model.log_alpha_h.clamp_(min=-3.0, max=3.0)    # alpha in [~0.05, ~20]
            model.log_alpha_rpm.clamp_(min=-3.0, max=3.0)

        # ---- logging ----
        if it % 100 == 0:
            print(f"[Iter {it}/{MAX_ITERS}]")
            for rec in per_ds_logs:
                print(
                    f"  - {rec['name']}: total={rec['total']:.4e} | "
                    f"pm={rec['pm']:.2e} ra={rec['ra']:.2e} rb={rec['rb']:.2e} "
                    f"cs={rec['cs']:.2e} st={rec['st']:.2e} end={rec['end']:.2e} hou={rec['hou']:.2e} | "
                    f"phy={rec['phy']:.2e} ic={rec['ic']:.2e} seq={rec['seq']:.2e} "
                    f"data={rec['data']:.2e}"
                )

            avg_total = total_loss_val / max(1, len(train_data_preloaded))
            ramp_pos  = 0 if not use_data else max(0, it - epoch_switched)
            print(
                f"  -> avg over train: {avg_total:.4e} | "
                f"ema_phy={ema_phy:.3e} (th={PHYS_THRESH:.3e}) | "
                f"use_data={use_data} | λ_data={lambda_data_this:.3e} | "
                f"ramp={ramp_pos}/{RAMP_EPOCHS} (Wmax={W_DATA_MAX})"
            )

            # (선택) alpha 디버그
            g_ah = None if model.log_alpha_h.grad is None else float(model.log_alpha_h.grad.detach().cpu())
            g_ar = None if model.log_alpha_rpm.grad is None else float(model.log_alpha_rpm.grad.detach().cpu())
            print(
                f"  -> log_alpha_h={model.log_alpha_h.item():+.3f} (alpha={float(torch.exp(model.log_alpha_h).detach().cpu()):.3f}) "
                f"grad={g_ah} | log_alpha_rpm={model.log_alpha_rpm.item():+.3f} grad={g_ar}"
            )

    # Adam 체크포인트
    adam_ckpt_path = os.path.join(save_dir, "hybrid_model_adam_only.pt")
    torch.save(model.state_dict(), adam_ckpt_path)
    print(f"\nAdam 학습 완료. 중간 체크포인트 저장 -> {adam_ckpt_path}")

    # ===== plot losses (phys -> ic -> data) =====
    if len(loss_hist["iter"]) > 2:
        x = np.asarray(loss_hist["iter"], dtype=float)

        plt.figure()
        plt.plot(x, np.asarray(loss_hist["phys"], dtype=float), color='red', label="phys")
        plt.plot(x, np.asarray(loss_hist["data"], dtype=float), color='darkblue', label="data")
        plt.yscale("log")  # 손실 스케일 차이 크면 log가 보통 유리
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True, which="major", linestyle="--", linewidth=0.8)
        plt.grid(False, which="minor")
        plt.tight_layout()
        plt.savefig(os.path.join(loss_plot_dir, "loss_phys_ic_data.png"), dpi=200)
        plt.close()


    

    # ===== LBFGS 미세 조정 (Phase 2: phys + alpha only) =====
    print("\n====== 2단계: LBFGS(phys+alpha) 미세 조정을 시작합니다... ======")

    LBFGS_FINETUNE_STEPS = 2000   # ← 0 말고 실제 값으로!
    PHYS_THRESH_LBFGS    = PHYS_THRESH

    ema_phy_lbfgs        = 0.0
    use_data_lbfgs       = bool(use_data)
    epoch_switched_lbfgs = (1 if use_data else None)

    print(f"[LBFGS INIT] use_data(adam)={use_data} → use_data(lbfgs)={use_data_lbfgs} | th={PHYS_THRESH_LBFGS:.3e}")

    for i in range(1, LBFGS_FINETUNE_STEPS + 1):

        per_ds_logs = []
        total_loss_val = [0.0]   # closure 밖에서 보기 위한 trick

        def closure():
            model.train()  # ★ cuDNN RNN backward를 위해 필수
            # 또는 최소한 GRU만: model.gru_encoder.train()

            model.opt_lbfgs_phys.zero_grad(set_to_none=True)

            total = torch.zeros((), dtype=next(model.parameters()).dtype, device=DEVICE)  # (권장)



            for data in train_data_preloaded:
                model.set_T_amb(T_AMB_FIXED)
                dom_id_t = torch.tensor([data["domain_id"]], dtype=torch.long, device=DEVICE)

                seq = data["loss_sequence"].squeeze(0).detach().cpu().numpy()
                time_grid = np.arange(0, PREDICT_TIME + 1e-9, 1.0)

                q_mag = denorm_channel(seq[:, 0], "mag")
                q_rot = denorm_channel(seq[:, 1], "rot")
                q_sta = denorm_channel(seq[:, 2], "sta")
                q_cop = denorm_channel(seq[:, 3], "cop")
                LPM   = denorm_channel(seq[:, 4], "LPM")
                RPM   = denorm_channel(seq[:, 6], "RPM") if seq.shape[1] >= 7 else np.zeros_like(time_grid)

                model.set_q_profile(time_grid, q_mag, q_rot, q_cop, q_sta, LPM=LPM, RPM=RPM)

                df_gt    = data["df_gt"]
                t_gt_sparse = df_gt["time"].values          # ★ 추가
                Tpm_gt   = df_gt["pm_tmax"].values
                Tcoil_gt = df_gt["coil_tmax"].values

                loss_ds, dbg_ds = model.sequential_consistency_loss(
                    data["loss_sequence"],
                    time_s_np=time_grid,
                    gt_time_s_np=t_gt_sparse,               # ★ 추가
                    Tpm_gt=Tpm_gt,
                    Tch_gt=Tcoil_gt,
                    domain_id=dom_id_t,
                    λ_ic=1.0,
                    λ_seq=1.0,
                    λ_data=1.0,
                    λ_teacher=2.0,
                    teacher_stride=10
                )


                total = total + (loss_ds / len(train_data_preloaded))
                total_loss_val[0] += float(loss_ds.item())

                per_ds_logs.append({
                    "name": data["info"]["name"],
                    "phy":  float(dbg_ds.get("phy", 0.0)),
                    "seq":  float(dbg_ds.get("seq", 0.0)),
                    "ic":   float(dbg_ds.get("ic",  0.0)),
                    "data": float(dbg_ds.get("data",0.0)),
                })

            total.backward()

            # 🔧 alpha 안정화
            with torch.no_grad():
                model.log_alpha_h.clamp_(min=-3.0, max=3.0)
    

            return total

        # ---- LBFGS step ----
        model.opt_lbfgs_phys.step(closure)

        # ---- EMA & logging ----
        if len(per_ds_logs) > 0:
            phys_avg_lbfgs = sum(rec["phy"] + rec["seq"] + rec["ic"] for rec in per_ds_logs) / len(per_ds_logs)
            ema_phy_lbfgs = 0.98 * ema_phy_lbfgs + 0.02 * phys_avg_lbfgs

        print(
            f"[LBFGS Iter {i}/{LBFGS_FINETUNE_STEPS}] "
            f"avg_total={total_loss_val[0]/len(train_data_preloaded):.3e} | "
            f"ema_phy={ema_phy_lbfgs:.3e} | "
            f"log_alpha_h={model.log_alpha_h.item():+.3f}"
        )
    
    finetuned_ckpt_path = os.path.join(save_dir, "hybrid_model_final_finetuned.pt")
    torch.save(model.state_dict(), finetuned_ckpt_path)
    print(f"미세 조정된 최종 모델 저장 -> {finetuned_ckpt_path}")


    # ===== 최종 검증 (Direct + Rollout) =====
    print("\n====== 5단계: 최종 모델 검증 (Ground Truth와 비교) ======")
    validation_model = HybridModel()
    validation_model.load_state_dict(torch.load(finetuned_ckpt_path, map_location=DEVICE))
    validation_model.eval()

    summary_rows = []
    for data in all_data_preloaded:
        dataset_info     = data["info"]
        loss_seq_tensor  = data["loss_sequence"]

        validation_model.set_T_amb(T_AMB_FIXED)

        seq = loss_seq_tensor.squeeze(0).detach().cpu().numpy()
        time_grid = np.arange(0, PREDICT_TIME + 1e-9, 1.0)

        q_mag = denorm_channel(seq[:, 0], "mag")
        q_rot = denorm_channel(seq[:, 1], "rot")
        q_sta = denorm_channel(seq[:, 2], "sta")
        q_cop = denorm_channel(seq[:, 3], "cop")
        LPM   = denorm_channel(seq[:, 4], "LPM")
        RPM   = denorm_channel(seq[:, 6], "RPM") if seq.shape[1] >= 7 else np.zeros_like(time_grid)
        Tamb  = np.full_like(time_grid, T_AMB_FIXED, dtype=np.float64)

        validation_model.set_q_profile(time_grid, q_mag, q_rot, q_cop, q_sta, LPM=LPM, RPM=RPM)

        dom_id_t = torch.tensor([dataset_info["domain_id"]], dtype=torch.long, device=DEVICE)

        # 예측: Direct + Rollout
        time_axis = np.unique(np.sort(df_gt["time"].values))

        Tpm_pred,  Tcoil_pred = validation_model.predict(loss_seq_tensor, time_axis, domain_id=dom_id_t, enforce_ic=True)

        Tpm_roll, Tcs_roll, Tend_roll = validation_model.predict_rollout(
            loss_seq_tensor, time_axis, domain_id=dom_id_t, enforce_ic=True
)
        
        print("predict direct coil head/tail:",
            Tcoil_pred[:5].tolist(),
            Tcoil_pred[-5:].tolist())

        # GT
# GT
        # [수정] 현재 루프의 데이터셋(data)에서 꺼내야 합니다.
        curr_gt = data["df_gt"]
        
        t_gt    = curr_gt["time"].to_numpy()
        pm_gt   = curr_gt["pm_tmax"].to_numpy()
        coil_gt = curr_gt["coil_tmax"].to_numpy()

        # Align
        pm_pred_plot   = np.interp(t_gt, time_axis, np.asarray(Tpm_pred))
        coil_pred_plot = np.interp(t_gt, time_axis, np.asarray(Tcoil_pred))
        pm_roll_plot   = np.interp(t_gt, time_axis, np.asarray(Tpm_roll))
        pm_roll_plot   = np.interp(t_gt, time_axis, np.asarray(Tpm_roll))
        coil_roll_plot = np.interp(t_gt, time_axis, np.asarray(Tend_roll))   # ✅ End-coil = Coil-Hot



        # overlap
        eps = 1e-6
        t_start = max(float(np.nanmin(t_gt)), float(np.nanmin(time_axis)))
        t_end   = min(float(np.nanmax(t_gt)), float(np.nanmax(time_axis)))
        mask_overlap = (t_gt >= t_start) & (t_gt <= t_end)
        t_eval       = t_gt[mask_overlap]
        pm_gt_eval   = pm_gt[mask_overlap]
        coil_gt_eval = coil_gt[mask_overlap]

        pm_pred_eval   = np.interp(t_eval, time_axis, np.asarray(Tpm_pred))
        coil_pred_eval = np.interp(t_eval, time_axis, np.asarray(Tcoil_pred))
        pm_roll_eval   = np.interp(t_eval, time_axis, np.asarray(Tpm_roll))
        coil_roll_eval = np.interp(t_eval, time_axis, np.asarray(Tend_roll)) # ✅ End-coil = Coil-Hot


        valid_pm_direct = np.isfinite(pm_gt_eval) & np.isfinite(pm_pred_eval) & (np.abs(pm_gt_eval) > eps)
        valid_co_direct = np.isfinite(coil_gt_eval) & np.isfinite(coil_pred_eval) & (np.abs(coil_gt_eval) > eps)
        valid_pm_roll   = np.isfinite(pm_gt_eval) & np.isfinite(pm_roll_eval) & (np.abs(pm_gt_eval) > eps)
        valid_co_roll   = np.isfinite(coil_gt_eval) & np.isfinite(coil_roll_eval) & (np.abs(coil_gt_eval) > eps)

        def smape_pct(y, yhat):
            den = np.abs(yhat) + np.abs(y) + eps
            return float(np.mean(200.0 * np.abs(yhat - y) / den))

        # %err arrays
        pm_pct_err_direct = (100.0 * (pm_pred_eval[valid_pm_direct] - pm_gt_eval[valid_pm_direct]) / np.abs(pm_gt_eval[valid_pm_direct])
                             if valid_pm_direct.any() else np.array([]))
        co_pct_err_direct = (100.0 * (coil_pred_eval[valid_co_direct] - coil_gt_eval[valid_co_direct]) / np.abs(coil_gt_eval[valid_co_direct])
                             if valid_co_direct.any() else np.array([]))
        pm_pct_err_roll = (100.0 * (pm_roll_eval[valid_pm_roll] - pm_gt_eval[valid_pm_roll]) / np.abs(pm_gt_eval[valid_pm_roll])
                           if valid_pm_roll.any() else np.array([]))
        co_pct_err_roll = (100.0 * (coil_roll_eval[valid_co_roll] - coil_gt_eval[valid_co_roll]) / np.abs(coil_gt_eval[valid_co_roll])
                           if valid_co_roll.any() else np.array([]))

        # metrics
        pm_mape_direct = float(np.mean(np.abs(pm_pct_err_direct))) if pm_pct_err_direct.size else np.nan
        co_mape_direct = float(np.mean(np.abs(co_pct_err_direct))) if co_pct_err_direct.size else np.nan
        pm_mape_roll   = float(np.mean(np.abs(pm_pct_err_roll)))   if pm_pct_err_roll.size else np.nan
        co_mape_roll   = float(np.mean(np.abs(co_pct_err_roll)))   if co_pct_err_roll.size else np.nan

        pm_smape_direct = smape_pct(pm_gt_eval[valid_pm_direct], pm_pred_eval[valid_pm_direct]) if valid_pm_direct.any() else np.nan
        co_smape_direct = smape_pct(coil_gt_eval[valid_co_direct], coil_pred_eval[valid_co_direct]) if valid_co_direct.any() else np.nan
        pm_smape_roll   = smape_pct(pm_gt_eval[valid_pm_roll], pm_roll_eval[valid_pm_roll]) if valid_pm_roll.any() else np.nan
        co_smape_roll   = smape_pct(coil_gt_eval[valid_co_roll], coil_roll_eval[valid_co_roll]) if valid_co_roll.any() else np.nan

        # ===== 플롯 (True + Direct + Rollout) =====
        ds_type = "Test" if "test" in dataset_info["name"] else "Train"

        plt.figure(figsize=(8, 6))
        plt.plot(t_gt, pm_gt, 'orange', label="True PM", linewidth=2)
        plt.plot(t_gt, pm_pred_plot, 'green', label="Direct", linewidth=2)
        plt.plot(t_gt, pm_roll_plot, 'blue', label="Rollout", linewidth=2)
        plt.title(f"PM Temp ({ds_type} - {dataset_info['name']}, Tamb={T_AMB_FIXED:.1f}°C)")
        plt.xlabel("Time (s)"); plt.ylabel("Temp (°C)")
        plt.ylim(10,300); plt.grid(True); plt.legend()
        plt.savefig(os.path.join(save_dir, f"finetuned_PM_{dataset_info['name']}.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(t_gt, coil_gt, 'orange', label="True Coil-Hot", linewidth=2)
        plt.plot(t_gt, coil_pred_plot, 'green', label="Direct", linewidth=2)
        plt.plot(t_gt, coil_roll_plot, 'blue', label="Rollout", linewidth=2)
        plt.title(f"Coil-Hot Temp ({ds_type} - {dataset_info['name']}, Tamb={T_AMB_FIXED:.1f}°C)")
        plt.xlabel("Time (s)"); plt.ylabel("Temp (°C)")
        plt.ylim(10, 300); plt.grid(True); plt.legend()
        plt.savefig(os.path.join(save_dir, f"finetuned_CoilHot_{dataset_info['name']}.png"), dpi=150)
        plt.close()

        # %error plot (Direct + Rollout)
        plt.figure(figsize=(8, 3.6))
        if valid_pm_direct.any():
            plt.plot(t_eval[valid_pm_direct], pm_pct_err_direct, label="PM % err (Direct)", linewidth=2)
        if valid_pm_roll.any():
            plt.plot(t_eval[valid_pm_roll], pm_pct_err_roll, label="PM % err (Rollout)", linewidth=2, linestyle="--")
        if valid_co_direct.any():
            plt.plot(t_eval[valid_co_direct], co_pct_err_direct, label="Coil % err (Direct)", linewidth=2)
        if valid_co_roll.any():
            plt.plot(t_eval[valid_co_roll], co_pct_err_roll, label="Coil % err (Rollout)", linewidth=2, linestyle="--")
        plt.axhline(0, ls="-", lw=1)
        plt.title(f"Percent Errors | {dataset_info['name']} | Overlap {t_start:.1f}–{t_end:.1f}s")
        plt.xlabel("Time (s)"); plt.ylabel("Error (%)")
        plt.grid(True); plt.legend()
        plt.savefig(os.path.join(save_dir, f"errors_pct_{dataset_info['name']}.png"), dpi=150)
        plt.close()

        # ===== CSV 저장 (overlap 구간) =====
        df_out = pd.DataFrame({
            "time_s_overlap": t_eval,
            "pm_gt": pm_gt_eval,
            "pm_pred_direct": pm_pred_eval,
            "pm_pred_roll": pm_roll_eval,
            "coil_gt": coil_gt_eval,
            "coil_pred_direct": coil_pred_eval,
            "coil_pred_roll": coil_roll_eval,
        })
        csv_path = os.path.join(save_dir, f"errors_table_{dataset_info['name']}.csv")
        df_out.to_csv(csv_path, index=False)

        summary_rows.append({
            "dataset": dataset_info["name"],
            "type": ds_type,
            "Tamb_degC": T_AMB_FIXED,
            "PM_MAPE_direct_%": pm_mape_direct,
            "Coil_MAPE_direct_%": co_mape_direct,
            "PM_sMAPE_direct_%": pm_smape_direct,
            "Coil_sMAPE_direct_%": co_smape_direct,
            "PM_MAPE_rollout_%": pm_mape_roll,
            "Coil_MAPE_rollout_%": co_mape_roll,
            "PM_sMAPE_rollout_%": pm_smape_roll,
            "Coil_sMAPE_rollout_%": co_smape_roll,
            "overlap_t_start_s": t_start,
            "overlap_t_end_s": t_end,
        })

        print(f"{dataset_info['name']} 완료")

    summary_df = pd.DataFrame(summary_rows).sort_values(by="dataset").reset_index(drop=True)
    summary_csv_path = os.path.join(save_dir, f"errors_summary_all_conditions_{script_name}.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print("요약 저장 완료 ->", summary_csv_path)
    print(summary_df)
