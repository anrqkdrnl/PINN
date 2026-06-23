# `lptn_single18_2.py` 코드 줄별 해석 (코딩 초보용)

> 이 문서는 논문 *"Physics-Informed operator learning for electric motor thermal transients"* 의 구현 코드를 **한 줄 한 줄** 풀어 설명한 학습 자료입니다.
> 똑같이 반복되는 줄(예: 동일한 신경망 층 8개)은 한 번만 설명하고 "반복"이라고 표시했습니다.
> 파이썬/PyTorch를 처음 보는 사람도 따라올 수 있도록 용어를 풀어 썼습니다.

---

## 📚 먼저 알아둘 기본 개념 5가지

코드를 보기 전에 이것만 알아두면 훨씬 쉽습니다.

1. **`import`** : 다른 사람이 만든 도구 상자(라이브러리)를 가져오는 명령. `import torch` = 토치라는 도구 상자를 쓰겠다.
2. **`class` (클래스)** : 관련된 데이터와 기능을 묶은 "설계도". 예를 들어 `class GRUEncoder`는 GRU 인코더를 만드는 설계도. 실제로 만들면(인스턴스화) 객체가 됨.
3. **`def` (함수/메서드)** : 어떤 작업을 하는 "기능 묶음". `def forward(...)`는 "앞으로 계산을 진행해라"라는 기능.
4. **`self`** : 클래스 안에서 "나 자신"을 가리키는 말. `self.hidden_dim`은 "내가 가진 hidden_dim 값".
5. **텐서(tensor)** : 숫자들의 묶음(배열). 신경망은 전부 텐서로 계산. `[B, N, D]`는 "B개의 데이터 × N개의 시간 × D개의 항목" 모양이라는 뜻.

신경망 구조 한 줄 요약:
**입력 조건 → GRU로 압축 → Branch Net으로 계수 만들기 → Trunk Net으로 시간함수 만들기 → 둘을 곱해서 온도 예측 → 물리법칙과 비교해서 오차 줄이기**

---

## 1부. 파일 머리말 & 라이브러리 가져오기 (1~46줄)

```python
# -*- coding: utf-8 -*-
"""
GRU Encoder + DeepONet Hybrid Model (PI-DeepONet; LPM/T_amb 기반 동적 냉각 버전)
...
"""
```
- 1줄 `# -*- coding: utf-8 -*-` : 이 파일이 한글 등 유니코드를 쓴다는 표시(요즘은 없어도 됨).
- 2~19줄 `""" ... """` : **독스트링(docstring)**. 큰따옴표 3개로 감싼 설명문. 실행에는 영향 없고, 사람이 읽으라고 적은 메모. "이 코드는 GRU + DeepONet 하이브리드 모델이고, 냉각을 LPM(유량)·외기온도 기반으로 동적으로 처리한다"는 내용.

```python
import os
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import matplotlib.pyplot as plt
```
- `import os` : 파일 경로·폴더 다루는 도구.
- `import numpy as np` : 숫자 계산 도구(넘파이)를 `np`라는 짧은 이름으로 가져옴.
- `import torch` : 딥러닝 핵심 도구(파이토치).
- `import torch.nn as nn` : 신경망 부품(층, 활성화함수 등)을 `nn`으로.
- `from torch import optim` : 학습 알고리즘(Adam, L-BFGS 등)을 `optim`으로.
- `from ... import CosineAnnealingLR` : 학습률(보폭)을 코사인 곡선처럼 점점 줄이는 스케줄러.
- `import pandas as pd` : 표(엑셀/CSV) 데이터 다루는 도구를 `pd`로.
- `import matplotlib.pyplot as plt` : 그래프 그리는 도구를 `plt`로.

```python
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
```
- 이 블록은 **재현성(reproducibility)** 을 위한 것. 신경망은 난수를 쓰는데, "씨앗(seed)"을 43으로 고정하면 매번 같은 난수가 나와서 **실행할 때마다 같은 결과**가 나옴.
- `os.environ["PYTHONHASHSEED"] = "43"` : 파이썬 내부 해시 난수 고정.
- `np.random.seed(43)` : 넘파이 난수 고정.
- `torch.manual_seed(43)` : 파이토치(CPU) 난수 고정.
- `if torch.cuda.is_available():` : GPU가 있으면
- `torch.cuda.manual_seed_all(43)` : GPU 난수도 고정.
- `try ... except Exception: pass` : "혹시 오류가 나도 프로그램을 멈추지 말고 그냥 넘어가라"는 안전장치. (라이브러리가 없어도 죽지 않게)

---

## 2부. 전역 설정값 (49~87줄)

여기는 프로그램 전체에서 쓰는 **설정값들을 한곳에 모아둔 부분**. 대문자 변수는 보통 "바꾸지 않는 상수"라는 관례.

```python
DATA_DIR      = "/home/hye/Documents/single/loss"
DATA_DIR2     = "/home/hye/Documents/single/gt"
SAVE_DIR_BASE = "/home/hye/Documents/single"
```
- 데이터가 있는 폴더 경로들. `loss`=발열량 CSV 폴더, `gt`=정답(ground truth) 온도 CSV 폴더, `SAVE_DIR_BASE`=결과 저장할 폴더.

```python
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```
- 계산을 어디서 할지 결정. GPU가 있으면 `cuda:0`(첫 번째 GPU), 없으면 `cpu`. 삼항 연산("A if 조건 else B" = 조건이 참이면 A, 아니면 B).

```python
PREDICT_TIME  = 3600.0   # 예측할 총 시간 = 3600초 = 1시간
T_MAX         = 400      # 온도 상승 스케일 상수 (논문의 ΔT_max에 해당, 정규화용)
TAMB          = 20       # 외기(주변) 온도 20도
TIME_WEIGHT_ALPHA = 0    # 시간 가중치 강도 (0이면 끔)
```
- 모델이 0~3600초 구간의 온도를 예측. `T_MAX`는 신경망 출력(0~1 사이 정규화값)을 실제 온도차로 바꿀 때 곱하는 스케일.

```python
GRU_INPUT_DIM  = 7    # GRU에 들어가는 입력 항목 수 = 7개
GRU_HIDDEN_DIM = 64   # GRU 내부 기억 공간 크기 = 64
GRU_N_LAYERS   = 4    # GRU 층 수 = 4겹
LEARNING_RATE  = 0.001  # 학습률(한 번에 얼마나 크게 배울지)
MAX_ITERS      = 20000  # 학습 반복 횟수 = 2만 번
GRAD_CLIP      = 1.0    # 기울기 폭주 방지 한계값
```
- 신경망 크기·학습 관련 핵심 하이퍼파라미터(사람이 정하는 설정값).
- 입력 7개 = [자석손실, 회전자손실, 고정자손실, 구리손실, LPM, 외기온도, RPM].

```python
CONTROL_DIM  = 7        # 제어 입력 차원
H_NORM_REF   = 2000.0   # 열전달계수 h 정규화 기준값
```
- 보조 설정값.

```python
LOSS_MIN_MAX = {
    'mag':  (44.2, 162.7),
    'rot':  (7.0, 107.5),
    'cop':  (55.3, 1811.1),
    'sta':  (139.8, 1070.5),
    'LPM':  (0.0, 1.0),
    'RPM':  (0.0, 8000),
}
```
- **딕셔너리(dictionary)**: "이름:값" 쌍의 모음. 여기선 각 입력 항목의 **최솟값·최댓값** 범위를 저장. 나중에 데이터를 0~1 사이로 정규화(min-max scaling)할 때 사용.
- 예: 자석손실(`mag`)은 44.2W~162.7W 범위.

```python
torch.set_default_dtype(torch.float64)
```
- 모든 숫자를 **64비트 실수(double)** 로 계산하라는 설정. 보통은 32비트를 쓰지만, 물리 계산은 정밀도가 중요해서 64비트를 씀.

---

## 3부. GRU 인코더 클래스 (92~145줄)

> **GRU란?** 시간 순서가 있는 데이터(시퀀스)를 처리하는 신경망의 한 종류. 운전 조건을 순서대로 읽어서 하나의 압축된 "요약 벡터"로 만듦.

```python
class GRUEncoder(nn.Module):
    def __init__(self, input_dim=GRU_INPUT_DIM, hidden_dim=GRU_HIDDEN_DIM, n_layers=GRU_N_LAYERS):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers
```
- `class GRUEncoder(nn.Module):` : GRU 인코더 설계도 시작. `nn.Module`을 상속(=PyTorch 신경망의 기본 틀을 물려받음).
- `def __init__(self, ...):` : **생성자**. 객체를 처음 만들 때 자동 실행되는 초기화 함수. 괄호 안은 입력 설정값(기본값은 위에서 정한 7, 64, 4).
- `super().__init__()` : 부모(nn.Module)의 초기화를 먼저 실행. (필수 절차)
- `self.hidden_dim = hidden_dim` : 받은 값을 "내 것"으로 저장.
- `self.n_layers = n_layers` : 층 수를 저장.

```python
        self.gru_var = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
        self.gru_fix = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True)
```
- GRU를 **두 개** 만듦. `gru_var`=변동 운전조건용, `gru_fix`=고정 운전조건용. 도메인(상황)에 따라 다른 GRU를 씀.
- `batch_first=True` : 데이터 모양을 `[배치, 시간, 항목]` 순서로 쓰겠다는 설정.

```python
        self.h0_var = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim, dtype=torch.float64))
        self.h0_fix = nn.Parameter(torch.zeros(n_layers, 1, hidden_dim, dtype=torch.float64))
```
- GRU의 **초기 기억 상태(h0)**. `nn.Parameter`로 만들면 **학습 가능한 값**이 됨(처음엔 0이지만 학습하며 바뀜). "콜드 스타트(시작이 0이라 불안정)"를 완화하려는 장치.

```python
    def _make_h0(self, B: int, device, dtype, is_var: bool):
        base = self.h0_var if is_var else self.h0_fix
        return base.to(device=device, dtype=dtype).expand(self.n_layers, B, self.hidden_dim).contiguous()
```
- 초기 상태 h0를 만들어주는 보조 함수.
- `base = ... if is_var else ...` : 변동용이면 `h0_var`, 아니면 `h0_fix` 선택.
- `.to(...)` : 적절한 장치(GPU/CPU)와 자료형으로 변환.
- `.expand(...)` : 배치 크기(B)만큼 복제. `.contiguous()` : 메모리를 깔끔히 정렬(GRU가 요구).

```python
    def forward(self, x, domain_id: torch.Tensor, return_sequences: bool = False):
```
- `forward` : 신경망의 **실제 계산** 함수. 입력 `x`가 들어오면 출력을 내보냄.
  - `x` : 입력 데이터 `[B, N, D]` (배치 × 시간 × 항목)
  - `domain_id` : 각 데이터가 변동(0)인지 고정(1)인지 표시
  - `return_sequences` : 모든 시간의 결과를 줄지(True), 마지막 것만 줄지(False)

```python
        B, N, _ = x.shape
        device, dtype = x.device, x.dtype
        y_out = torch.zeros(B, N, self.hidden_dim, device=device, dtype=dtype)
```
- `B, N, _ = x.shape` : 입력 모양에서 배치 크기 B, 시간 길이 N을 꺼냄. `_`는 "안 쓸 값"이라는 관례.
- `device, dtype = ...` : 입력이 어느 장치/자료형인지 파악.
- `y_out = torch.zeros(...)` : 결과를 담을 빈 상자(0으로 채운 텐서)를 미리 만듦.

```python
        mask_var = (domain_id == 0)
        mask_fix = (domain_id == 1)
```
- **마스크**: 어떤 데이터가 변동(0)이고 고정(1)인지 참/거짓으로 표시한 목록.

```python
        if mask_var.any():
            x_var = x[mask_var]
            Bv    = x_var.size(0)
            h0    = self._make_h0(Bv, device, dtype, is_var=True)
            y_var, _ = self.gru_var(x_var, h0)
            y_out[mask_var] = y_var
```
- `if mask_var.any():` : 변동 데이터가 하나라도 있으면 실행.
- `x_var = x[mask_var]` : 변동 데이터만 골라냄.
- `Bv = x_var.size(0)` : 변동 데이터 개수.
- `h0 = self._make_h0(...)` : 변동용 초기 상태 준비.
- `y_var, _ = self.gru_var(x_var, h0)` : 변동 GRU로 계산. 결과 y_var와 마지막 상태(_, 안 씀)를 받음.
- `y_out[mask_var] = y_var` : 결과를 원래 자리에 넣음.

```python
        if mask_fix.any():
            x_fix = x[mask_fix]
            Bf    = x_fix.size(0)
            h0    = self._make_h0(Bf, device, dtype, is_var=False)
            y_fix, _ = self.gru_fix(x_fix, h0)
            y_out[mask_fix] = y_fix
```
- 위와 똑같은데 **고정 데이터용**(`gru_fix`)으로 처리.

```python
        if return_sequences:
            return y_out
        return y_out[:, -1, :]
```
- `return_sequences`가 참이면 모든 시간 결과 `[B,N,H]`를 반환.
- 아니면 `y_out[:, -1, :]` = 마지막 시간의 결과 `[B,H]`만 반환. (`-1`은 "맨 마지막")

---

## 4부. Branch Net 클래스 (149~171줄)

> **Branch Net이란?** DeepONet의 두 날개 중 하나. GRU가 압축한 운전조건을 받아서 "계수 벡터"를 만듦.

```python
class BN(nn.Module):
    def __init__(self, input_dim=GRU_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            ... (총 8개의 Linear+Tanh) ...
        )
        self._init_weights()
```
- `self.net = nn.Sequential(...)` : 여러 층을 **순서대로** 쌓은 신경망. `nn.Sequential`은 "이 순서대로 통과시켜라".
- `nn.Linear(input_dim, 64)` : **선형 층(완전연결층)**. 입력을 받아 64개 숫자로 변환. 신경망의 기본 계산 단위.
- `nn.Tanh()` : **활성화 함수**(하이퍼볼릭 탄젠트). 출력을 -1~1 사이로 눌러서 비선형성을 줌(이게 없으면 아무리 층을 쌓아도 직선밖에 못 그림).
- 이 패턴(`Linear → Tanh`)이 **8번 반복** = 8층짜리 깊은 신경망. 폭(뉴런 수)은 64로 일정. → 논문의 "8 linear layers, hidden width 64, tanh activation"과 정확히 일치.
- `self._init_weights()` : 가중치 초기값을 설정하는 함수 호출.

```python
    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
```
- `for layer in self.net:` : 신경망의 각 층을 하나씩 돌아봄.
- `if isinstance(layer, nn.Linear):` : 그 층이 선형 층이면(Tanh는 건너뜀)
- `xavier_uniform_(layer.weight)` : **자비에 초기화**. 가중치를 적절한 크기의 난수로 채워 학습이 잘 시작되게 함.
- `zeros_(layer.bias)` : 편향(bias)은 0으로 시작.

```python
    def forward(self, u):
        return self.net(u)
```
- 입력 `u`를 쌓아둔 신경망에 통과시켜 결과 반환. (이게 Branch Net의 실제 계산)

---

## 5부. Trunk Net 클래스들 — PM, COIL, Rotor, Stator (174~273줄)

> **Trunk Net이란?** DeepONet의 다른 날개. **시간 `t`** 를 입력받아 시간에 따른 "기저함수"를 만듦. 부품마다 따로 둠(PM=영구자석, COIL=코일, Rotor=회전자, Stator=고정자).

```python
class PM(nn.Module):
    def __init__(self, control_dim=CONTROL_DIM):
        super().__init__()
        in_dim = 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Tanh(),
            ... (총 8개) ...
        )
        self._init_weights()
```
- `in_dim = 1` : 입력이 **딱 1개(시간 t)**. → 논문에서 "각 trunk는 시간 t만 입력으로 받는다"고 한 부분.
- 나머지는 Branch Net과 **구조가 완전히 동일**(8층, 폭 64, Tanh, 자비에 초기화).

```python
    def forward(self, t_and_ctrl):
        return self.net(t_and_ctrl)
```
- 시간을 받아 신경망 통과 후 반환.

**`COIL`, `Rotor`, `Stator` 클래스(199~273줄)** 는 이름만 다를 뿐 **PM과 구조가 100% 똑같습니다.** (각각 코일/회전자/고정자 부품의 시간함수를 학습하기 위해 별도로 만든 것.) 부품마다 열적 시간상수가 다르기 때문에 따로 둡니다 → 논문의 "separate trunk networks for rotor, PM, coil, stator, housing".

```python
def _to_scalar(x: torch.Tensor, reduce: str = "mean") -> torch.Tensor:
    if not torch.is_tensor(x):
        return torch.as_tensor(x, dtype=torch.get_default_dtype())
    if x.ndim == 0:
        return x
    return x.mean() if reduce == "mean" else x.sum()
```
- 보조 함수: 어떤 값을 **하나의 숫자(스칼라)** 로 만들어줌.
- 텐서가 아니면 텐서로 바꾸고, 이미 0차원(숫자 하나)이면 그대로, 아니면 평균(mean) 또는 합(sum)으로 압축.

---

(다음 파트에서 핵심인 HybridModel 클래스로 이어집니다)

---

## 6부. HybridModel 클래스 — 전체 모델 조립 (285~309줄)

> 이 클래스가 **논문의 그림 4 전체 구조**를 하나로 합칩니다. 위에서 만든 GRU, Branch, Trunk들을 다 모으고, LPTN 물리 상수도 여기에 넣습니다.

```python
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = DEVICE
        self.gru_encoder = GRUEncoder().to(self.device)
        self.BN = BN(input_dim=GRU_HIDDEN_DIM).to(self.device)
```
- `self.device = DEVICE` : 계산 장치(GPU/CPU) 저장.
- `self.gru_encoder = GRUEncoder().to(self.device)` : 위에서 만든 GRU 인코더를 실제로 생성하고 장치에 올림. `.to(device)`는 "이 장치로 옮겨라".
- `self.BN = BN(...)` : Branch Net 생성.

```python
        self.ROTOR_A = Rotor().to(self.device)
        self.PM      = PM().to(self.device)
        self.ROTOR_B = Rotor().to(self.device)
        self.CS      = COIL().to(self.device)      # 코일+고정자 겹침 구간
        self.STATOR  = Stator().to(self.device)    # 순수 고정자
        self.END     = COIL().to(self.device)      # 엔드 와인딩(코일 끝부분)
```
- **노드별 Trunk Net**들을 생성. 논문 그림 3의 LPTN 노드들과 대응:
  - `ROTOR_A` : 회전자 A (회전자~자석 경계)
  - `PM` : 영구자석
  - `ROTOR_B` : 회전자 B (자석~코일 경계)
  - `CS` : 코일-고정자 겹침 영역 (Coil-Stator)
  - `STATOR` : 고정자 본체
  - `END` : 코일 끝(엔드 와인딩)

```python
        self.log_alpha_h   = nn.Parameter(torch.tensor(0.0, ...))
        self.log_alpha_rpm = nn.Parameter(torch.tensor(0.0, ...))
```
- **학습 가능한 보정계수**(논문 Eq. 23의 θ_h). `log_alpha_h`는 냉각수 열전달계수 h에 곱할 보정값을 로그 형태로 저장(처음엔 0 → exp(0)=1, 즉 보정 없음에서 시작). 학습하며 자동 조정됨.
- 로그를 쓰는 이유: `α = exp(θ)`로 하면 항상 양수가 보장됨.

---

## 7부. 옵티마이저 그룹 설정 (323~408줄)

> **옵티마이저란?** 신경망이 오차를 줄이도록 가중치를 조금씩 바꿔주는 "학습 엔진". 여기선 부품 그룹별로 따로 만듦.

```python
        params_seq = []
        params_seq += list(self.gru_encoder.parameters())
        params_seq += list(self.BN.parameters())
        params_pm = list(self.PM.parameters())
        params_rotor = []
        if hasattr(self, "ROTOR_A") and self.ROTOR_A is not None:
            params_rotor += list(self.ROTOR_A.parameters())
        ...
```
- `.parameters()` : 그 신경망이 학습할 모든 가중치를 가져옴.
- `params_seq` : GRU + Branch의 파라미터 묶음.
- `params_pm`, `params_rotor`, ... : 부품별로 파라미터를 따로 모음.
- `if hasattr(self, "ROTOR_A")` : "ROTOR_A라는 게 있으면"이라는 안전 확인. (없을 때 오류 방지)

```python
        def _uniq(params):
            seen = set()
            out = []
            for p in params:
                if p is None: continue
                if not isinstance(p, torch.Tensor): continue
                if not p.requires_grad: continue
                if id(p) in seen: continue
                out.append(p)
                seen.add(id(p))
            return out
```
- **중복 제거 함수**. 같은 파라미터가 두 번 들어가면 오류가 나므로, 중복·학습 불가능한 것을 걸러냄.
- `seen = set()` : 이미 본 것을 기록할 집합.
- `if not p.requires_grad: continue` : 학습 대상이 아니면 건너뜀.
- `if id(p) in seen: continue` : 이미 본 것이면 건너뜀.

```python
        LR_SEQ     = getattr(self, "LR_SEQ", LEARNING_RATE)
        LR_PM      = getattr(self, "LR_PM", 1e-4)
        ...
```
- 그룹마다 **학습률(보폭)** 을 따로 설정. `getattr(self, "LR_PM", 1e-4)` = "LR_PM이 있으면 그걸 쓰고, 없으면 0.0001을 써라".

```python
        self.opt_seq     = optim.Adam(params_seq,    lr=LR_SEQ)
        self.opt_pm      = optim.Adam(params_pm,     lr=LR_PM) if len(params_pm) > 0 else None
        ...
        self.optimizers = [self.opt_seq, self.opt_pm, ...]
        self.optimizers = [o for o in self.optimizers if o is not None]
```
- `optim.Adam(...)` : **Adam 옵티마이저** 생성(가장 널리 쓰는 학습 알고리즘).
- `if len(params_pm) > 0 else None` : 파라미터가 있을 때만 만들고, 없으면 None.
- `self.optimizers = [...]` : 모든 옵티마이저를 리스트로 모음(나중에 한 번에 호출하려고).
- 마지막 줄: None인 것은 제외.

```python
        self.scheduler_seq = CosineAnnealingLR(self.opt_seq, T_max=MAX_ITERS, eta_min=1e-7)
```
- **학습률 스케줄러**. 학습이 진행될수록 보폭을 코사인 곡선처럼 점점 줄여서(0.001 → 거의 0) 정밀하게 수렴하게 함.

```python
        self.opt_lbfgs_phys = optim.LBFGS(
            params_lbfgs, max_iter=4, history_size=20, line_search_fn="strong_wolfe"
        )
```
- **L-BFGS 옵티마이저**. Adam으로 대략 학습한 뒤 마지막에 정밀하게 다듬는 2차 최적화 기법. → 논문의 "Adam → L-BFGS" 2단계 전략.

---

## 8부. 물리 상수 정의 — LPTN의 핵심 (426~607줄)

> 여기가 논문 그림 3의 **열회로망(LPTN)을 숫자로 구현**한 부분. 열용량(C), 열저항(R), 면적(A), 길이(L), 열전도율(k) 등을 정의.

```python
        self.register_buffer('C_pm',     torch.tensor(1000.34,  dtype=dtype))
        self.register_buffer('C_rotorA', torch.tensor(2980.39, dtype=dtype))
        self.register_buffer('C_rotorB', torch.tensor(658.49,  dtype=dtype))
        self.register_buffer('C_CS',     torch.tensor(4447.28, dtype=dtype))
        self.register_buffer('C_stator', torch.tensor(1754.46, dtype=dtype))
        self.register_buffer('C_end',    torch.tensor(3111.60, dtype=dtype))
        self.register_buffer('C_housing', torch.tensor(28823.0319, dtype=dtype))
```
- `register_buffer(...)` : **학습되지 않는 고정 상수**를 모델에 등록(파라미터와 달리 학습 중 안 바뀜).
- `C_xxx` : 각 부품의 **열용량 [J/K]** (열을 얼마나 저장할 수 있는지). 부피 × 밀도 × 비열로 계산한 값. 논문 Table 2의 물성치 기반.

```python
        self.register_buffer('A_ra_pm',    torch.tensor(0.0332,   dtype=dtype))
        self.register_buffer('A_pm_rb',    torch.tensor(0.027064, dtype=dtype))
        ...
        self.register_buffer('A_end_surf', torch.tensor(0.014334,  dtype=dtype))
```
- `A_xxx` : 부품 사이 **접촉 면적 [m²]**. 열이 통과하는 면적. 예: `A_ra_pm`=회전자A와 자석 사이 면적.

```python
        self.register_buffer('k_air',   torch.tensor(0.026, dtype=dtype))
        self.register_buffer('k_steel', torch.tensor(42.0,  dtype=dtype))
        self.register_buffer('k_pm',    torch.tensor(8.9,   dtype=dtype))
        self.register_buffer('k_cu',    torch.tensor(401.0, dtype=dtype))
```
- `k_xxx` : **열전도율 [W/m·K]** (그 재료가 열을 얼마나 잘 전달하는지). 공기 0.026(나쁨), 강철 42, 자석 8.9, 구리 401(아주 좋음). 논문 Table 2와 일치.

```python
        self.register_buffer('L_rotorA', torch.tensor(27.79e-3, dtype=dtype))
        self.register_buffer('L_pm',     torch.tensor(6.07e-3,  dtype=dtype))
        ...
```
- `L_xxx` : 각 구간의 **두께/길이 [m]**. `27.79e-3` = 0.02779m = 27.79mm.

```python
        self.register_buffer('delta_gap',  torch.tensor(0.6e-3, dtype=dtype))
        self.register_buffer('h_conv0',    torch.tensor(250.0,  dtype=dtype))
        self.register_buffer('h_conv_exp', torch.tensor(0.5,    dtype=dtype))
        self.register_buffer('h_air', torch.tensor(1700.0, dtype=dtype))
```
- 에어갭(공기 틈) 관련 상수. `delta_gap`=공기 틈 두께, `h_conv0`/`h_conv_exp`=회전에 의한 대류 열전달 공식의 계수, `h_air`=엔드코일 공기 대류 계수.

```python
        def _R_cond(L, k, A, eps=1e-12):
            return (L / (k * A + eps)).to(dtype)
```
- **전도 열저항 계산 함수**. 공식 `R = L/(k·A)` (논문 Eq. 6). 두께가 두껍고 면적이 작고 전도율이 낮을수록 저항이 큼. `eps`는 0으로 나누기 방지용 아주 작은 수.

```python
        def _h_conv_rot_from_rpm(rpm):
            rpm_n = torch.clamp(rpm / (self.RPM_NORM_REF + 1e-12), min=0.0)
            return self.h_conv0 * (rpm_n ** self.h_conv_exp)
```
- 회전속도(RPM)로부터 회전 대류 열전달계수를 계산. RPM이 빠를수록 공기가 휘저어져 냉각이 잘 됨.
- `torch.clamp(..., min=0.0)` : 음수가 되지 않게 0으로 자름.
- `** self.h_conv_exp` : 0.5 제곱(제곱근).

```python
        V_cu_CS = 4.899e-4
        V_st_CS = 7.64e-4
        f_cu = V_cu_CS / (V_cu_CS + V_st_CS)
        f_st = 1.0 - f_cu
        k_CS_eff_val = 401.0 * f_cu + 42.0 * f_st  # ≈ 182 W/mK
        self.register_buffer('k_CS_eff', torch.tensor(k_CS_eff_val, dtype=dtype))
```
- CS 영역은 구리와 강철이 섞여 있어서 **유효 열전도율**을 부피 비율로 평균. `f_cu`=구리 비율, `f_st`=강철 비율. 결과 약 182.

```python
        self.register_buffer("R_ra_pm", _R_cond(self.L_rotorA, self.k_steel, self.A_ra_pm))
        self.register_buffer("R_pm_rb", _R_cond(self.L_pm, self.k_pm, self.A_pm_rb))
        self.register_buffer("R_rb_cs", _R_cond(self.L_rotorB, self.k_steel, self.A_rb_cs))
        self.register_buffer("R_cs_st", _R_cond(self.L_CS, self.k_CS_eff, self.A_cs_st))
        self.register_buffer("R_st_ch", _R_cond(self.L_stator, self.k_steel, self.A_st_ch))
```
- 위 `_R_cond` 함수로 **부품 사이 열저항(R)** 들을 계산해 등록. 이게 LPTN 회로의 "저항"들. 논문 그림 3의 R1~R9에 해당.

```python
        self.A_pm = torch.tensor(0.00145, ...)     # PM 대류 면적
        self.alpha_pm_h = torch.tensor(0.5, ...)   # PM 냉각 강도 스케일
        self.register_buffer('A_coil', torch.tensor(0.076))
        self.use_dynamic_h = True
        self.h_profile     = None
        self.LPM_profile   = None
        self.RPM_profile   = None
        self.q_profile     = None
        self.T_amb = torch.tensor(20.0, ...)
```
- `use_dynamic_h = True` : 냉각수 h를 LPM에서 동적으로 계산하겠다는 스위치.
- `h_profile`, `LPM_profile` 등 = None : 아직 시간별 프로파일이 안 들어온 빈 상태.
- `T_amb` : 외기온도 20도 고정.

---

## 9부. 보조 메서드들 (610~801줄)

```python
    def set_T_amb(self, t_amb):
        self.T_amb = torch.tensor(20.0, ...)
```
- 외기온도를 설정하는 함수인데, 이 실험에선 **항상 20도로 고정**(입력값 t_amb를 무시).

```python
    def _h_gap_eff(self, RPM_t):
        ...
        h_cond = self.k_air / (self.delta_gap + 1e-12)
        rpm_n = RPM_t / (self.RPM_NORM_REF.to(dtype) + 1e-12)
        h_conv_rot = self.h_conv0.to(dtype) * torch.clamp(rpm_n, min=0.0) ** self.h_conv_exp.to(dtype)
        return h_cond + h_conv_rot
```
- 에어갭의 유효 열전달계수 = 전도분(`h_cond`) + 회전대류분(`h_conv_rot`). 공기 틈을 통한 열전달을 모델링.

```python
    def _h_from_LPM_tcool_rpm_dual(self, LPM_t, Tcool_t, RPM_t=None):
```
- **이 함수가 논문의 핵심(Eq. 15~23)** : 냉각수 유량(LPM)으로부터 열전달계수 h를 계산. 줄별로 보면:

```python
        LPM = LPM_t.view(-1).clamp(min=0.0)
        eps = 1e-6
        coolant_on = (LPM > eps).to(dtype=dtype)
```
- `coolant_on` : 유량이 0보다 크면 1(냉각 켬), 0이면 0(냉각 끔). 논문 Eq. 16의 on-off 게이트.

```python
        Q_total = LPM * (1e-3 / 60.0)
        Q_m = Q_total
```
- LPM(분당 리터)을 **초당 m³** 단위로 변환. 논문 Eq. 15와 동일. 모든 유량을 하나의 채널로 보냄(단일채널 가정).

```python
        Dh_m = torch.tensor(... 9.6e-3 ...)      # 수력직경
        Aflow_m = torch.tensor(... 9.6e-5 ...)   # 유동 단면적
```
- 채널 형상. 논문 Eq. 17 (직사각형 12mm×8mm → Dh≈9.6mm).

```python
        rho = ... 997.0     # 밀도
        mu  = ... 0.00089   # 점성
        kf  = ... 0.60      # 열전도율
        cp  = ... 4180.0    # 비열
        Pr  = (cp * mu / (kf + 1e-12)).clamp(min=1e-6)
```
- 냉각수(물) 물성치. 논문 Table 4. `Pr`=프란틀 수(논문 Eq. 18).

```python
        Vm = (Q_m / (Aflow_m + 1e-12)).clamp(min=0.0)   # 유속
        Re_m = (rho * Vm * Dh_m / (mu + 1e-12)).clamp(min=0.0)  # 레이놀즈 수
```
- 유속(V)과 **레이놀즈 수(Re)** 계산. 논문 Eq. 18. Re는 흐름이 느린지(층류) 빠른지(난류) 판단하는 무차원 수.

```python
        def _Nu(Re):
            Nu_lam = torch.full_like(Re, 4.36)
            Nu_tur = 0.023 * (Re.clamp(min=1.0) ** 0.8) * (Pr ** 0.4)
            return torch.where(Re < 2300.0, Nu_lam, Nu_tur)
        Nu_m = _Nu(Re_m)
```
- **누셀트 수(Nu)** 계산. Re가 2300 미만이면 층류(Nu=4.36), 이상이면 난류(Dittus-Boelter 식, 논문 Eq. 20). `torch.where(조건, A, B)`=조건 참이면 A, 아니면 B.

```python
        h_m = (Nu_m * kf / (Dh_m + 1e-12)).clamp(min=0.0)
```
- 누셀트 수로부터 **열전달계수 h** 계산. 논문 Eq. 22 (`h = Nu·k/Dh`).

```python
        if hasattr(self, "log_alpha_h"):
            alpha = torch.exp(self.log_alpha_h.to(dtype))
            h_m = alpha * h_m
        h_eff = coolant_on * h_m
        return h_eff.view(-1, 1)
```
- **학습 가능한 보정계수 α를 곱함**(논문 Eq. 23의 `α_h = exp(θ_h)`). 그리고 냉각이 켜진 경우에만 유효(`coolant_on` 곱하기). 이게 최종 h.

```python
    def set_q_profile(self, time_s, q_mag, q_rot, q_cop, q_sta, h=None, LPM=None, RPM=None):
        self.q_profile = (np.asarray(time_s), ...)
        self.h_profile   = (...) if h   is not None else None
        ...
```
- 시간별 **발열량/유량 프로파일을 모델에 저장**하는 함수. 나중에 임의 시각의 값을 보간해서 꺼내 쓸 수 있게 함.

```python
    def _interp_q_at(self, t_norm_vec):
        ts, qm, qr, qc, qs = self.q_profile
        t_phys = (t_norm_vec.detach().cpu().numpy().reshape(-1) * PREDICT_TIME)
        qm_t = np.interp(t_phys, ts, qm)
        ...
```
- **보간(interpolation) 함수**. 정규화된 시간(0~1)을 실제 시간(초)으로 바꾼 뒤, 저장된 프로파일에서 그 시각의 발열량·유량 값을 `np.interp`(선형 보간)로 계산해 꺼냄.

```python
    def output(self, u, t_feat):
        if u.size(0) == t_feat.size(0):
            return (u * t_feat).sum(dim=1, keepdim=True)
        if u.size(0) == 1:
            u_rep = u.expand(t_feat.size(0), -1)
            return (u_rep * t_feat).sum(dim=1, keepdim=True)
        raise ValueError(...)
```
- **DeepONet의 핵심 연산**: Branch 출력(`u`)과 Trunk 출력(`t_feat`)을 **곱해서 더함(내적)**. 논문 Eq. 2의 `G(U)(Y) = Σ bk·zk`. 이게 최종 온도(정규화값)를 만드는 부분.
- `(u * t_feat).sum(dim=1)` : 원소별 곱 후 합. `keepdim=True`는 모양 유지.

```python
    def _node_temp_and_dt(self, bout, trunk, t_vec):
        tout = trunk(t_vec)
        S = self.output(bout, tout).view(-1, 1)
        dSdt = torch.autograd.grad(S.sum(), t_vec, create_graph=True)[0].view(-1, 1)
        dT_rel = S * T_MAX
        ddTdt  = dSdt * T_MAX / PREDICT_TIME
        return dT_rel, ddTdt
```
- 한 노드의 **온도와 그 시간미분(dT/dt)** 을 계산.
- `torch.autograd.grad(...)` : **자동미분**. 신경망 출력 S를 시간 t로 미분(이게 PINN의 마법! 논문 Eq. 8). 물리방정식에 dT/dt가 필요하기 때문.
- `dT_rel = S * T_MAX` : 정규화 출력을 실제 온도차로 환원(논문 Eq. 7).
- `ddTdt` : 온도의 시간변화율(체인룰로 실제 시간 단위로 환산).


---

## 10부. 손실 함수 `sequential_consistency_loss` (877~1674줄) ⭐가장 중요

> 이 함수가 **논문 3.2.2절의 모든 손실항**을 계산합니다. 모델이 "물리적으로 맞고 + 데이터와도 맞게" 학습되도록 오차를 만드는 핵심.

### 10-1. 입력과 준비 (877~931줄)

```python
    def sequential_consistency_loss(self, loss_sequence, time_s_np,
        Tpm_gt=None, Tcs_gt=None, Tch_gt=None, domain_id=None,
        λ_ic=1, λ_seq=1.0, λ_data=1.0, gt_time_s_np=None,
        λ_teacher=5.0, teacher_stride=5):
```
- 입력 설명:
  - `loss_sequence` : 발열량 입력 시퀀스
  - `time_s_np` : 시간축
  - `Tpm_gt`, `Tch_gt` : 정답 온도(자석, 코일) — `gt`=ground truth
  - `λ_ic`, `λ_seq`, `λ_data`, `λ_teacher` : 각 손실항의 **가중치**(λ=람다). 어떤 손실을 얼마나 중요하게 볼지.

```python
        def _v(x):
            if x is None: return None
            x = ... .to(device=device, dtype=model_dtype)
            if x.dim() == 1: return x.view(-1, 1)
            return x
```
- 보조 함수: 입력을 적절한 장치·자료형·모양으로 정리. 1차원이면 세로 벡터 `[N,1]`로 바꿈.

```python
        def _clamp_R(R):
            return torch.clamp(R, min=1e-9)
```
- 열저항이 0이나 음수가 되지 않게 최소 1e-9로 자름(0으로 나누기 방지).

### 10-2. Branch 계산 (932~967줄)

```python
        h_seq = self.gru_encoder(loss_sequence, domain_id=domain_id, return_sequences=True).squeeze(0)
        Nh_full = int(h_seq.size(0))
        u_seq_full = h_seq
```
- GRU로 입력 시퀀스를 인코딩(모든 시간 결과 받음). `.squeeze(0)`=불필요한 첫 차원 제거.

```python
        t_np = np.asarray(time_s_np, ...).reshape(-1)
        t_np = np.unique(t_np); t_np.sort()
        Nt = int(len(t_np))
```
- 시간축 정리: 중복 제거(`unique`) + 정렬(`sort`).

```python
        dt_base = float(PREDICT_TIME) / float(Nh_full - 1)
        idx = np.rint(t_np / dt_base).astype(np.int64)
        idx = np.clip(idx, 0, Nh_full - 1)
        u_seq = u_seq_full[idx]
        bout  = self.BN(u_seq)
```
- 원하는 시각에 해당하는 GRU 출력을 골라(`idx`로 인덱싱), **Branch Net에 통과**시켜 계수 벡터 `bout` 생성.

```python
        t = torch.as_tensor(np.clip(t_np / PREDICT_TIME, 0.0, 1.0), ...).view(-1, 1)
        t.requires_grad_(True)
```
- 시간을 0~1로 정규화(논문 Eq. 4). `requires_grad_(True)` : **이 시간으로 미분할 거니까 추적해라**(자동미분 준비). PINN의 필수 단계.

### 10-3. Trunk 계산 + 자동미분 (969~1029줄)

```python
        phi_pm = self.PM(t).to(model_dtype)
        phi_ra = self.ROTOR_A(t)...
        phi_rb = self.ROTOR_B(t)...
        phi_cs  = self.CS(t)...
        phi_st  = self.STATOR(t)...
        phi_end = self.END(t)...
        phi_hou = ... (HOUSING 없으면 STATOR로 대체)
```
- 각 부품의 Trunk Net에 시간 t를 넣어 **기저함수 φ(phi)** 들을 계산.

```python
        def dot(b, phi):
            return (b * phi).sum(dim=1, keepdim=True)
        S_pm  = dot(bout, phi_pm)
        S_ra  = dot(bout, phi_ra)
        ... (모든 노드)
```
- Branch(`bout`)와 각 Trunk(`phi`)를 내적해서 각 노드의 **정규화 온도 S** 계산(논문 Eq. 2).

```python
        dSdt_pm  = torch.autograd.grad(S_pm.sum(),  t, create_graph=True)[0]
        ... (모든 노드)
```
- 각 노드 온도를 시간으로 **자동미분**해서 dS/dt 계산(논문 Eq. 8). `create_graph=True`는 "미분의 미분도 가능하게 그래프를 유지하라".

```python
        dTpm  = S_pm  * T_MAX     # 정규화값 → 실제 온도차 [K]
        ...
        scale_dt = (T_MAX / PREDICT_TIME)
        dTpm_dt  = dSdt_pm  * scale_dt   # 실제 dT/dt
        ...
```
- 정규화된 값들을 **실제 물리 단위(온도차, 온도변화율)** 로 환산.

### 10-4. 발열량·열전달계수 준비 (1031~1162줄)

```python
        q_mag, q_rot, q_cop, q_sta, h_t, LPM_t, Tamb_t, RPM_t = self._interp_q_at(t.view(-1))
        q_mag = _v(q_mag); ... 
```
- 각 시각의 발열량(자석/회전자/구리/고정자)과 LPM/RPM을 보간으로 꺼냄.

```python
        T_pm  = T_amb_vec + dTpm
        T_ra  = T_amb_vec + dTra
        ... (모든 노드)
```
- **절대온도 = 외기온도 + 온도차**. 논문 Eq. 7.

```python
        factor = 1.0 + alpha_cu * (T_cs - T_ref_vec)
        factor = torch.clamp(factor, min=0.7, max=2.0)
        q_cop_eff = q_cop * factor
```
- 구리 손실 **온도 보정**: 구리는 뜨거워지면 저항이 커져 발열이 늘어남. 그걸 반영. 0.7~2배로 제한.

```python
        V_cs  = 4.899e-4; V_end = 9.024e-4
        f_cs  = V_cs / (V_cs + V_end); f_end = V_end / (V_cs + V_end)
        q_cop_cs  = q_cop_eff * f_cs
        q_cop_end = q_cop_eff * f_end
```
- 구리 발열을 부피 비율로 **CS 영역과 엔드코일에 나눠줌**.

```python
        if getattr(self, "use_dynamic_h", False):
            if h_t is not None:
                h_dyn = torch.clamp(h_t, min=1e-6)
            else:
                h_dyn = self._h_from_LPM_tcool_rpm_dual(LPM_t.view(-1), T_cool_vec.view(-1), RPM_t.view(-1))
                h_dyn = torch.clamp(h_dyn, min=1e-6)
```
- **동적 냉각수 h 계산**: 위 8부의 함수를 호출해 LPM으로부터 h를 만듦.

```python
        R_conv = 1.0 / (h_dyn * A_st_use + 1e-9)
        R_st_cool_raw = _clamp_R(R_wall + R_conv)
        R_st_cool_t = R_st_cool_raw
```
- 냉각수 대류 열저항 `R = 1/(h·A)` (논문 Eq. 6) + 벽 전도저항을 더해 **고정자→냉각수 총 열저항** 계산.

```python
        R_end_amb = _clamp_R(1.0 / (h_air * A_end_surf + 1e-9))
        R_pm_amb  = _clamp_R(1.0 / (h_air * A_pm_surf  + 1e-9))
```
- 엔드코일·자석에서 공기로 빠지는 대류 열저항.

```python
        rpm_n = RPM_t / (rpm_ref + 1e-12)
        h_conv_rot = h_conv0 * torch.clamp(rpm_n, min=0.0) ** h_conv_exp
        h_gap_eff = h_cond + h_conv_rot
        R_rb_cs_t = _clamp_R(1.0 / (h_gap_eff * A_rb_cs + 1e-9))
```
- 에어갭(회전자~코일 공기 틈) 열저항을 RPM에 따라 동적으로 계산.

### 10-5. 물리 잔차(Physics Residual) — PINN의 심장 (1192~1252줄)

> **잔차(residual)란?** 물리방정식을 "= 0" 형태로 만들었을 때, 신경망 예측을 넣으면 얼마나 0에서 벗어나는지. 0에 가까울수록 물리법칙을 잘 지킨 것. 논문 Eq. 9.

```python
        res_pm = (
            C_pm * dTpm_dt          # 열용량 × 온도변화율 (저장된 열)
            - q_mag                  # 발열량 빼기
            + (T_pm - T_ra) / R_pm_ra   # 회전자A로 나가는 열
            + (T_pm - T_rb) / R_pm_rb   # 회전자B로 나가는 열
            + (T_pm - T_amb_vec) / R_pm_amb  # 공기로 나가는 열
        )
```
- 자석 노드의 에너지 보존식. 논문 Eq. 5를 그대로 코드로 옮긴 것:
  **(열용량×변화율) = 발열 − 주변으로 빠져나간 열**. 이게 0이어야 물리적으로 맞음.

```python
        res_ra = ( C_ra * dTra_dt - q_rot_a + (T_ra - T_pm)/R_pm_ra + (T_ra - T_cs)/R_ra_cs_t )
        res_rb = ( C_rb * dTrb_dt - q_rot_b + ... )
        res_cs = ( C_cs * dTcs_dt - q_cop_cs + ... )   # 코일-고정자
        res_st = ( C_st * dTst_dt - q_sta + ... )      # 고정자
        res_end = ( C_end * dTend_dt - q_cop_end + ... )  # 엔드코일
        res_hou = ( C_hou * dTh_dt - ... )             # 하우징
```
- 나머지 6개 노드도 똑같은 방식으로 각자의 에너지 보존식 잔차를 만듦. 각 노드가 **어떤 이웃과 연결되는지**가 논문 그림 3의 회로 구조 그대로.

```python
        if R_st_cool_t is not None:
            res_st = res_st + coolant_on * (T_st - T_cool_vec) / R_st_cool_t
```
- 냉각이 켜져 있으면 고정자 방정식에 **냉각수로 빠지는 열**을 추가. (논문의 "coolant heat removal" 항)

```python
        loss_pm  = (res_pm**2).mean()  / 1e7
        loss_ra  = (res_ra**2).mean()  / 1e6
        ... (각 노드)
        loss_phy = loss_pm + loss_ra + loss_rb + loss_cs + loss_st + loss_end + loss_hou
```
- 각 잔차를 **제곱→평균**(MSE, 논문 Eq. 10)하고, 노드마다 크기가 다르므로 `/1e7` 같은 값으로 나눠 스케일을 맞춤. 모두 더해 **물리 손실 `loss_phy`** 완성.

### 10-6. 초기조건 손실 (1254~1280줄)

```python
        t0 = torch.zeros((1, 1), ...)
        b0 = bout[0:1]
        def S0(phi_fn):
            return (b0 * phi_fn(t0)...).sum(dim=1, keepdim=True)
        S0_pm = S0(self.PM); ...
        dT0 = (S0_pm.squeeze()*T_MAX).pow(2) + ... (모든 노드)
        loss_ic = (λ_ic * dT0)
```
- **초기조건(IC) 손실**(논문 Eq. 12): t=0일 때 온도차가 0이어야 함(시작 온도=외기온도). t=0에서의 예측값을 제곱해서, 0이 아니면 벌점. 시작점이 어긋나면 전체가 틀어지므로 중요.

### 10-7. 평활화 정규화 (1282~1306줄)

```python
        c_diff = bout[1:] - bout[:-1]
        per_step_energy = (c_diff.pow(2).sum(dim=1) / (Cb + 1e-8))
        ...
        L_smooth = lambda_smooth * (gate[:Lm] * per_step_energy[:Lm]).mean()
```
- **평활화(smoothness) 손실**(논문 Eq. 14): Branch 계수가 시간에 따라 너무 급격히 변하지 않게 함. `bout[1:] - bout[:-1]`=연속한 시각 사이 차이. 차이가 크면 벌점 → 매끄러운 곡선 유도.

### 10-8. 데이터 손실 (1308~1360줄)

```python
        def _sparse_mse(T_pred_1d, gt_times_np, gt_vals_np):
            ...
            idx = torch.where(dist_lo <= dist_hi, idx_lo, idx_hi)
            pred = T_pred_1d[idx]
            return ((pred - gt_v) ** 2).mean()
```
- **데이터 손실**(논문 Eq. 11): 실제 정답(GT) 온도가 있는 시각에서, 예측과 정답의 차이를 제곱평균. GT 시각에 가장 가까운 예측값을 찾아 비교.

```python
        if Tpm_gt is not None:
            mse_pm = _sparse_mse(T_pm.view(-1), gt_time_s_np, Tpm_gt)
            if mse_pm is not None:
                data_loss = data_loss + mse_pm / 0.2e3
        ...
```
- 자석·코일 각각 데이터 손실을 계산해 더함. `/0.2e3`=스케일 조정.

### 10-9. 순차 일관성 손실 (1362~1454줄)

```python
        dts = torch.as_tensor(np.diff(t_np), ...).view(-1, 1)
        Tpm_k = T_pm[:-1]; ... (현재 시각 온도들)
        dTpm_rhs = ( qmag_k - (Tpm_k - Tra_k)/R_pm_ra - ... ) / C_pm
        ...
        Tpm_e  = Tpm_k + dts * dTpm_rhs
        ...
        seq_loss = ((Tpm_e - T_pm[1:])**2).mean() + ... (모든 노드)
```
- **순차 일관성(sequential consistency) 손실**(논문 Eq. 13): 전진 오일러법으로 "현재 온도 + dt×변화율 = 다음 온도"가 성립하는지 검사.
  - `dts = np.diff(t_np)` : 시간 간격.
  - `dTpm_rhs` : 물리방정식 우변(변화율).
  - `Tpm_e = Tpm_k + dts * dTpm_rhs` : 오일러법으로 예측한 다음 온도.
  - `seq_loss` : 이 오일러 예측과 신경망이 직접 낸 다음 온도(`T_pm[1:]`)가 일치하는지 비교. 시간적으로 매끄럽고 오차 누적을 막음.

### 10-10. Teacher-rollout 증류 (1456~1615줄)

```python
        if (λ_teacher is not None) and (float(λ_teacher) > 0.0) and (N >= 2):
            with torch.no_grad():
                ...
                for k in range(dts_use.size(0)):
                    dTpm_rhs = (...)
                    Tpm_r = Tpm_r + dt * dTpm_rhs
                    ...
                    Tpm_roll.append(Tpm_r)
```
- **Teacher 롤아웃**: 순수 물리방정식만으로 처음부터 끝까지 한 스텝씩 적분해서 "선생님(teacher) 온도 곡선"을 만듦.
- `with torch.no_grad():` : 이 부분은 미분 추적 안 함(선생님은 학습 대상이 아니라 목표일 뿐).
- `for k in range(...)` : 시간을 한 스텝씩 전진하며 온도 갱신(전진 오일러).

```python
            L_teacher = (
                ((Tpm_direct_u - Tpm_teacher)**2).mean() + ... (모든 노드)
            )
```
- **증류(distillation) 손실**: 신경망이 직접 낸 예측(student)이 물리 적분으로 만든 선생님 곡선을 따라하도록 유도. 물리적으로 안정적인 곡선을 학습.

### 10-11. 손실 합산 (1617~1674줄)

```python
        loss_phy_s  = _to_scalar(loss_phy, "mean")
        loss_ic_s   = _to_scalar(loss_ic, "mean")
        ...
        # total = loss_phy_s + loss_ic_s + seq_loss_s + data_loss_s + L_smooth_s   ← 주석 처리됨
        total = data_loss_s
```
- ⚠️ **주의 깊게 볼 부분**: 원래는 모든 손실을 합치게 되어 있는데(주석 처리된 부분), 현재 활성화된 줄은 `total = data_loss_s` **하나만**. 즉 지금 이 버전은 **데이터 손실만으로 학습**하도록 임시 설정되어 있음. (실험 중 바꾼 흔적으로 보임. 물리 손실을 다시 켜려면 주석을 해제해야 함.)

```python
        dbg = { "phy": ..., "ic": ..., "data": ..., ... }
        return total, dbg
```
- `dbg` : 디버깅용 정보 묶음(각 손실 값, 예측 온도 등)을 딕셔너리로 정리.
- `return total, dbg` : 최종 손실과 디버그 정보를 돌려줌.


---

## 11부. 예측 함수 `predict` — Direct 방식 (1680~1753줄)

> 학습이 끝난 뒤 **신경망으로 한 번에 온도 곡선을 예측**하는 함수(논문에서 빠른 추론에 해당).

```python
    def predict(self, loss_sequence, time_axis, domain_id=None, enforce_ic: bool = True):
        self.eval()
```
- `self.eval()` : 모델을 **평가 모드**로 전환(학습용 동작 끔).

```python
        t_phys = torch.as_tensor(time_axis, ...).view(-1)
        N = int(t_phys.numel())
        t_norm = (t_phys / PREDICT_TIME).clamp(0.0, 1.0).view(-1, 1)
```
- 예측할 시각들을 준비하고 0~1로 정규화.

```python
        h_full = self.gru_encoder(loss_sequence, domain_id=domain_id, return_sequences=True).squeeze(0)
        h_seq = h_full.index_select(0, idx)
        bout = self.BN(h_seq)
```
- GRU → 시각별 인덱싱 → Branch Net으로 계수 생성. (학습 때와 동일한 과정)

```python
        phi_pm = self.PM(t_norm)...
        phi_hot = self.CS(t_norm)...
        dTpm = self.output(bout, phi_pm).squeeze(1) * T_MAX
        dTh  = self.output(bout, phi_hot).squeeze(1) * T_MAX
```
- 자석은 PM trunk로, 코일-hot은 CS trunk로 온도차 예측. (코드 주석대로 코일은 CS로 통일, END 안 씀)

```python
        Tamb_vec = torch.full((N,), 20.0, ...)
        Tpm  = (dTpm + Tamb_vec).detach().cpu().numpy()
        Thot = (dTh  + Tamb_vec).detach().cpu().numpy()
        if enforce_ic and N > 0:
            Tpm[0] = 20.0
            Thot[0] = 20.0
        return Tpm, Thot
```
- 절대온도 = 온도차 + 20도. `.detach().cpu().numpy()`=계산 그래프 분리 후 넘파이 배열로 변환.
- `enforce_ic` : 시작값을 강제로 20도로 맞춤.
- **자석온도(Tpm)와 코일온도(Thot)를 반환.**

---

## 12부. 예측 함수 `predict_rollout` — Rollout 방식 (1764~2098줄)

> Direct가 "신경망이 통째로 예측"이라면, Rollout은 **물리방정식을 한 스텝씩 직접 적분**해서 곡선을 만드는 방식. 두 방식을 비교해 모델의 물리적 타당성을 검증.

```python
    def predict_rollout(self, loss_sequence, time_axis, domain_id=None, enforce_ic=True,
                        dt_internal=1.0, rollout_end_time=None, return_full=False):
```
- `dt_internal=1.0` : 내부 적분 시간 간격 1초.
- `rollout_end_time` : 끝 시간(None이면 자동).

```python
        time_axis = np.unique(time_axis); time_axis = np.sort(time_axis)
        if time_axis.size < 2:
            raise ValueError("time_axis must have at least 2 points.")
```
- 시간축 정리 및 검증(최소 2개 점 필요).

```python
        t_internal = np.arange(t0, t_end + 1e-12, dt_internal, ...)
        sample_idx = np.searchsorted(t_internal, time_axis, side="left")
```
- **촘촘한 내부 시간격자** 생성(1초 간격). 입력 시각이 듬성듬성해도 1초 단위로 적분한 뒤 필요한 점만 샘플링.

```python
        with torch.no_grad():
            Tamb0 = 20.0; Tcool0 = 30.0
            dTpm = torch.tensor(0.0, ...)   # 자석 온도차 초기값 0
            dTrb = ...; dTra = ...; dTcs = ...; dTst = ...; dTend = ...; dTh = ...
```
- `with torch.no_grad():` : 미분 추적 안 함(추론만).
- 각 노드의 온도차를 0에서 시작(=초기온도 20도).

```python
            C_pm = self.C_pm...; R_pm_rb = ...; R_cs_st = ...; ...
```
- 학습 때 정한 열용량·열저항 상수들을 가져옴.

```python
            for i in range(t_internal.size - 1):
                t_curr = float(t_internal[i])
                dt = float(t_internal[i + 1] - t_internal[i])
```
- **시간 루프**: 1초씩 전진하며 온도를 갱신.

```python
                q_mag, q_rot, q_cop, q_sta, h_t, LPM_t, Tamb_t, RPM_t = self._interp_q_at(t_norm)
```
- 현재 시각의 발열량·유량을 보간으로 꺼냄.

```python
                is_flow_off = (float(LPM_s.item()) <= LPM_EPS)
                R_end_amb_eff = torch.tensor(1e12, ...) if is_flow_off else R_end_amb
```
- 유량이 0이면 냉각 끔. 냉각 없을 때 엔드코일-공기 저항을 사실상 무한대(1e12)로 → 그 경로 차단.

```python
                if not is_flow_off:
                    h_dyn = self._h_from_LPM_tcool_rpm_dual(...)
                R_st_cool = ... 1.0 / (h_dyn * A_st_use + 1e-12) ...
```
- 냉각이 켜져 있으면 동적 h로 고정자-냉각수 저항 계산.

```python
                dTpm_dt = ( q_mag_s - (Tpm-Trb)/R_pm_rb - (Tpm-Tra)/R_ra_pm - (Tpm-T_amb)/R_pm_amb ) / C_pm
                dTrb_dt = ( q_rot_s - ... ) / C_rb
                dTra_dt = ( q_rot_s - ... ) / C_ra
                dTcs_dt = ( q_cop_cs - ... ) / C_cs
                dTst_dt = ( base_st - ... ) / C_st
                dTend_dt = ( q_cop_end - ... ) / C_end
                dTh_dt = ( ... ) / C_hou
```
- **각 노드의 온도변화율(dT/dt)** 을 LPTN 방정식으로 계산. (10-5의 잔차식과 같은 물리식)

```python
                dTpm  = dTpm  + dt_t * dTpm_dt
                dTrb  = dTrb  + dt_t * dTrb_dt
                ... (모든 노드)
```
- **전진 오일러 적분**: 새 온도차 = 기존 온도차 + 시간간격 × 변화율. 한 스텝 전진.

```python
                Tpm_dense[i + 1]  = float(Tamb0 + dTpm.item())
                Tcs_dense[i + 1]  = float(Tamb0 + dTcs.item())
                Tend_dense[i + 1] = float(Tamb0 + dTend.item())
```
- 매 스텝의 절대온도를 저장.

```python
            if enforce_ic and Tpm_dense.size > 0:
                Tpm_dense[0] = Tamb0; ...
```
- 시작값을 20도로 강제.

```python
        if return_full:
            return t_internal, Tpm_dense, Tcs_dense, Tend_dense
        Tpm_out  = Tpm_dense[sample_idx]
        Tcs_out  = Tcs_dense[sample_idx]
        Tend_out = Tend_dense[sample_idx]
        return Tpm_out, Tcs_out, Tend_out
```
- 촘촘한 결과에서 요청한 시각만 골라 반환(자석, CS, 엔드코일 온도).

---

## 13부. 데이터 로딩 보조 함수들 (2106~2232줄)

```python
def min_max_scale(data, min_val, max_val, eps=1e-8):
    return (data - min_val) / (max_val - min_val + eps)
```
- **정규화 함수**: 값을 0~1 사이로 변환. `(값-최소)/(최대-최소)`.

```python
def denorm_channel(q_norm, key):
    qmin, qmax = LOSS_MIN_MAX[key]
    return q_norm * (qmax - qmin) + qmin
```
- **역정규화 함수**: 0~1 값을 원래 물리값으로 되돌림(정규화의 반대).

```python
def load_gt_with_time(gt_path: str) -> pd.DataFrame:
    df = pd.read_csv(gt_path)
    df.columns = df.columns.astype(str).str.replace("\ufeff", "", ...).str.strip()
    ...
```
- 정답(GT) CSV를 읽는 함수. CSV 형식이 제각각일 수 있어서(구분자가 `,`인지 `;`인지, BOM 문자, 컬럼명이 'Time'/'t'/'sec' 등) **여러 예외 상황을 처리**해서 'time' 컬럼을 확실히 찾아냄.

```python
def load_and_prepare_sequence(loss_csv_path, t_amb_fill=None):
    df = pd.read_csv(loss_csv_path)
    time_grid = np.arange(0, PREDICT_TIME + 1e-9, 1.0)
    required = ["time", "magnet_loss", "rotor_loss", "copper_loss", "stator_loss"]
    for c in required:
        if c not in df.columns:
            raise KeyError(...)
```
- 발열량 CSV를 읽어 **신경망 입력 시퀀스로 변환**하는 함수.
- `time_grid = np.arange(0, 3600, 1.0)` : 0~3600초를 1초 간격으로(총 3601개 점).
- 필수 컬럼이 없으면 오류.

```python
    q_mag_raw = np.interp(time_grid, t_src, df["magnet_loss"].values)
    ... (4개 손실 보간)
    if "LPM" in df.columns:
        LPM_raw = np.interp(time_grid, t_src, df["LPM"].values)
    else:
        LPM_raw = np.full_like(time_grid, 0.0, ...)
    Tamb_raw = np.full_like(time_grid, 20.0, ...)
```
- 각 손실을 1초 간격으로 보간. LPM이 있으면 보간, 없으면 0. 외기온도는 20 고정.

```python
    q_mag = min_max_scale(q_mag_raw, LOSS_MIN_MAX["mag"][0], LOSS_MIN_MAX["mag"][1])
    ... (정규화)
    seq7 = np.stack([q_mag, q_rot, q_sta, q_cop, LPM_n, Tamb_n, RPM_n], axis=1)...
    return torch.from_numpy(seq7).unsqueeze(0).to(DEVICE)
```
- 모든 채널을 정규화한 뒤 **7개 채널을 묶어** `[1, 3601, 7]` 모양의 텐서로 만들어 반환.

```python
def load_legacy_single_gru_ckpt(model, ckpt_path, ...):
```
- 예전 버전(단일 GRU)으로 저장된 모델을 새 버전(듀얼 GRU)으로 불러오는 호환 함수. (지금 실행엔 거의 안 쓰임)

```python
def _infer_domain_id(name: str) -> int:
    name_lower = name.lower()
    return 0 if "variable" in name_lower else 1
```
- 데이터셋 이름에 "variable"이 있으면 0(변동), 없으면 1(고정)로 도메인 판단.

---

## 14부. 메인 실행부 시작 (2299~2435줄)

> `if __name__ == "__main__":` 아래는 **이 파일을 직접 실행할 때만** 작동하는 부분(실제 프로그램의 시작점).

```python
if __name__ == "__main__":
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = os.path.join(SAVE_DIR_BASE, script_name)
    os.makedirs(save_dir, exist_ok=True)
    T_AMB_FIXED = 20.0
```
- `if __name__ == "__main__":` : "이 파일이 메인으로 실행되면"이라는 파이썬 관례.
- 결과 저장 폴더 이름을 스크립트 이름으로 만들고 생성(`exist_ok=True`=이미 있어도 오류 안 냄).

```python
    datasets_info = [
        {"name": "Case1_0(train)", "loss_csv": ..., "gt_merged": ...},
        {"name": "Case1_1(train)", ...},
        {"name": "Case3_0(train)", ...},
        {"name": "Case2_0(train)", ...},
    ]
```
- **학습용 데이터셋 4개**. 각각 발열량 CSV(`loss_csv`)와 정답 CSV(`gt_merged`) 경로를 딕셔너리로. (논문에선 8개 학습이라 했는데, 이 버전은 4개로 줄여 실험 중인 듯)

```python
    test_dataset_info = [
        {"name": "Case1_0.2(test)", ...},
        ... (총 14개)
    ]
```
- **테스트용 데이터셋들**. Case1/2/3 × 여러 유량(0.2~1.0 LPM). 학습에 안 쓴 조건으로 일반화 성능 평가(논문의 generalization 검증).

```python
    all_datasets_info = datasets_info + test_dataset_info
```
- 학습+테스트 합치기.

```python
    all_data_preloaded = []
    for ds_info in all_datasets_info:
        loss_seq_tensor = load_and_prepare_sequence(ds_info["loss_csv"], t_amb_fill=T_AMB_FIXED)
        assert loss_seq_tensor.dim() == 3 and loss_seq_tensor.size(-1) == 7, ...
        df_gt = load_gt_with_time(ds_info["gt_merged"])
        df_gt = df_gt[df_gt["time"] <= PREDICT_TIME].copy()
```
- **모든 데이터를 미리 메모리에 로딩**(매번 파일 읽으면 느리니까).
- `assert ...` : 모양이 `[1, T, 7]`이 맞는지 검증(아니면 멈춤).
- `df_gt[df_gt["time"] <= PREDICT_TIME]` : 3600초 이내 데이터만 사용.

```python
        if not (df_gt["time"].values == 0).any():
            df0 = pd.DataFrame([{"time": 0.0, "pm_tmax": 20.0, "coil_tmax": 20.0}])
            df_gt = pd.concat([df0, df_gt], ...).sort_values("time")...
```
- 정답에 t=0 데이터가 없으면 "0초=20도" 행을 직접 추가(초기조건 보장).

```python
        all_data_preloaded.append({
            "info": ds_info, "loss_sequence": loss_seq_tensor,
            "df_gt": df_gt, "t_train": df_gt["time"].values, "domain_id": dom_id,
        })
```
- 로딩한 데이터를 리스트에 차곡차곡 저장.

```python
    train_data_preloaded = [d for d in all_data_preloaded if "train" in d["info"]["name"]]
    test_data_preloaded  = [d for d in all_data_preloaded if "test"  in d["info"]["name"]]
    assert len(train_data_preloaded) > 0, "학습 데이터가 없습니다."
```
- 이름에 'train'/'test'가 있는지로 학습/테스트 분리. **리스트 컴프리헨션**(`[... for ... if ...]`=조건 맞는 것만 골라 새 리스트).

```python
    model = HybridModel()
    model.eval()
```
- **모델 생성!** 위에서 설계한 HybridModel을 실제로 만듦.

---

## 15부. Adam 학습 루프 (2438~2647줄) ⭐

```python
    PHYS_THRESH   = 1e9
    EMA_MOMENTUM  = 0.98
    W_DATA_MAX    = 1.0
    RAMP_EPOCHS   = 3000
    GRAD_CLIP     = 1.0
    ema_phy        = 0.0
    use_data       = True
    epoch_switched = 0
```
- 학습 전략용 설정값. `use_data=True`=데이터 손실 사용, `RAMP_EPOCHS`=가중치를 서서히 올리는 구간.

```python
    def _ramp_weight(iter_idx, start_iter, ramp_iters, w_max):
        if (start_iter is None) or (iter_idx < start_iter): return 0.0
        r = min(1.0, float(iter_idx - start_iter) / max(1, ramp_iters))
        return float(w_max * r)
```
- **램프(ramp) 가중치 함수**: 데이터 손실 가중치를 0에서 시작해 서서히 최대값까지 끌어올림(갑자기 큰 가중치를 주면 학습이 불안정해지므로).

```python
    for it in range(MAX_ITERS):
        model.train()
        lambda_data_this = _ramp_weight(it, epoch_switched, RAMP_EPOCHS, W_DATA_MAX) if use_data else 0.0
```
- **메인 학습 반복**(2만 번). `model.train()`=학습 모드. 이번 반복의 데이터 손실 가중치 계산.

```python
        for opt in model.optimizers:
            opt.zero_grad(set_to_none=True)
```
- 모든 옵티마이저의 **기울기 초기화**(이전 반복의 기울기를 지움. 안 하면 누적됨).

```python
        for data in train_data_preloaded:
            model.set_T_amb(T_AMB_FIXED)
            dom_id_t = torch.tensor([data["domain_id"]], ...)
            seq = data["loss_sequence"].squeeze(0).detach().cpu().numpy()
            q_mag = denorm_channel(seq[:, 0], "mag"); ... (역정규화)
            model.set_q_profile(time_grid, q_mag, q_rot, q_cop, q_sta, LPM=LPM, RPM=RPM)
```
- **각 학습 데이터에 대해**: 발열량을 원래 물리값으로 되돌려 모델에 프로파일로 설정.

```python
            loss_ds, dbg_ds = model.sequential_consistency_loss(
                data["loss_sequence"], time_s_np=time_grid,
                gt_time_s_np=t_gt_sparse, Tpm_gt=pm_gt_sparse, Tch_gt=coil_gt_sparse,
                domain_id=dom_id_t, λ_ic=1.0, λ_seq=1.0,
                λ_data=float(lambda_data_this), λ_teacher=2.0, teacher_stride=10)
```
- **손실 계산!** 10부의 손실 함수를 호출. 각 손실의 가중치를 넘김.

```python
            (loss_ds / len(train_data_preloaded)).backward()
            total_loss_val += float(loss_ds.item())
```
- `.backward()` : **역전파**! 손실을 각 가중치로 미분해 "어느 방향으로 바꿔야 손실이 줄지" 계산. 데이터 개수로 나눠 평균(기울기 누적).

```python
        torch.nn.utils.clip_grad_norm_(model.gru_encoder.parameters(), GRAD_CLIP)
        torch.nn.utils.clip_grad_norm_(model.BN.parameters(), GRAD_CLIP)
        ... (각 부품별)
        torch.nn.utils.clip_grad_norm_([model.log_alpha_h, model.log_alpha_rpm], GRAD_CLIP * 5.0)
```
- **기울기 클리핑**: 기울기가 너무 커지면(폭주) 학습이 망가지므로 최대 크기를 1.0으로 제한. α는 좀 더 널널하게(×5).

```python
        for opt in model.optimizers:
            opt.step()
        model.scheduler_seq.step()
```
- `opt.step()` : **실제로 가중치 업데이트**(배운 방향으로 한 걸음).
- `scheduler_seq.step()` : 학습률을 코사인 스케줄대로 줄임.

```python
        with torch.no_grad():
            model.log_alpha_h.clamp_(min=-3.0, max=3.0)
            model.log_alpha_rpm.clamp_(min=-3.0, max=3.0)
```
- α 보정계수가 너무 커지거나 작아지지 않게 -3~3으로 제한(exp 하면 약 0.05~20배).

```python
        if it % 100 == 0:
            print(f"[Iter {it}/{MAX_ITERS}]")
            ... (손실 출력)
```
- 100번마다 진행 상황 출력(`%`=나머지 연산, it가 100의 배수일 때).

```python
    adam_ckpt_path = os.path.join(save_dir, "hybrid_model_adam_only.pt")
    torch.save(model.state_dict(), adam_ckpt_path)
```
- Adam 학습 끝나면 모델 저장(`.pt` 파일). `state_dict()`=모델의 모든 가중치.

```python
    if len(loss_hist["iter"]) > 2:
        plt.figure()
        plt.plot(x, ..., color='red', label="phys")
        plt.plot(x, ..., color='darkblue', label="data")
        plt.yscale("log"); ...
        plt.savefig(...)
```
- **손실 곡선 그래프**를 그려 저장(물리 손실=빨강, 데이터 손실=파랑). y축은 로그 스케일.

---

## 16부. L-BFGS 미세조정 (2669~2769줄)

```python
    LBFGS_FINETUNE_STEPS = 2000
    for i in range(1, LBFGS_FINETUNE_STEPS + 1):
        def closure():
            model.train()
            model.opt_lbfgs_phys.zero_grad(set_to_none=True)
            total = torch.zeros((), ...)
            for data in train_data_preloaded:
                ... (손실 계산)
                total = total + (loss_ds / len(train_data_preloaded))
            total.backward()
            return total
        model.opt_lbfgs_phys.step(closure)
```
- **L-BFGS 미세조정 2000회**. L-BFGS는 같은 계산을 여러 번 해야 해서 `closure`(다시 계산하는 함수)를 넘겨줌.
- Adam으로 대략 학습한 모델을 더 정밀하게 다듬음. → 논문의 2단계 최적화.

```python
    finetuned_ckpt_path = os.path.join(save_dir, "hybrid_model_final_finetuned.pt")
    torch.save(model.state_dict(), finetuned_ckpt_path)
```
- 최종 모델 저장.

---

## 17부. 최종 검증 & 결과 저장 (2772~2952줄)

```python
    validation_model = HybridModel()
    validation_model.load_state_dict(torch.load(finetuned_ckpt_path, ...))
    validation_model.eval()
```
- 저장한 최종 모델을 새로 불러와 검증 준비.

```python
    for data in all_data_preloaded:
        ...
        Tpm_pred, Tcoil_pred = validation_model.predict(loss_seq_tensor, time_axis, ...)
        Tpm_roll, Tcs_roll, Tend_roll = validation_model.predict_rollout(loss_seq_tensor, time_axis, ...)
```
- **모든 데이터에 대해** Direct 예측과 Rollout 예측을 둘 다 수행.

```python
        curr_gt = data["df_gt"]
        t_gt = curr_gt["time"].to_numpy()
        pm_gt = curr_gt["pm_tmax"].to_numpy()
        coil_gt = curr_gt["coil_tmax"].to_numpy()
        pm_pred_plot = np.interp(t_gt, time_axis, np.asarray(Tpm_pred))
        ...
```
- 정답을 꺼내고, 예측을 정답 시각에 맞춰 보간(비교하려고 시각 정렬).

```python
        def smape_pct(y, yhat):
            den = np.abs(yhat) + np.abs(y) + eps
            return float(np.mean(200.0 * np.abs(yhat - y) / den))
        pm_mape_direct = float(np.mean(np.abs(pm_pct_err_direct))) ...
```
- **오차 지표 계산**: MAPE(평균 절대 백분율 오차), sMAPE(대칭 버전). 논문 결과표(Table 6)에 해당.

```python
        plt.figure(figsize=(8, 6))
        plt.plot(t_gt, pm_gt, 'orange', label="True PM", ...)
        plt.plot(t_gt, pm_pred_plot, 'green', label="Direct", ...)
        plt.plot(t_gt, pm_roll_plot, 'blue', label="Rollout", ...)
        plt.savefig(os.path.join(save_dir, f"finetuned_PM_{...}.png"), ...)
```
- **자석 온도 그래프** 저장: 정답(주황) vs Direct(초록) vs Rollout(파랑). 논문 그림 7·8 같은 비교 그래프.

```python
        plt.figure(...)
        plt.plot(t_gt, coil_gt, 'orange', label="True Coil-Hot", ...)
        ... 
        plt.savefig(... f"finetuned_CoilHot_{...}.png" ...)
```
- **코일 온도 그래프**도 똑같이 저장.

```python
        plt.figure(...)
        plt.plot(..., pm_pct_err_direct, label="PM % err (Direct)", ...)
        ...
        plt.savefig(... f"errors_pct_{...}.png" ...)
```
- **백분율 오차 그래프** 저장.

```python
        df_out = pd.DataFrame({
            "time_s_overlap": t_eval, "pm_gt": pm_gt_eval,
            "pm_pred_direct": pm_pred_eval, "pm_pred_roll": pm_roll_eval, ...
        })
        df_out.to_csv(csv_path, index=False)
```
- 시각별 정답·예측값을 **CSV로 저장**.

```python
        summary_rows.append({
            "dataset": ..., "type": ds_type, "PM_MAPE_direct_%": ...,
            "Coil_MAPE_direct_%": ..., "PM_MAPE_rollout_%": ..., ...
        })
```
- 각 데이터셋의 오차 지표를 요약 행에 추가.

```python
    summary_df = pd.DataFrame(summary_rows).sort_values(by="dataset")...
    summary_df.to_csv(summary_csv_path, index=False)
    print("요약 저장 완료 ->", summary_csv_path)
    print(summary_df)
```
- 모든 데이터셋의 오차를 **하나의 요약표(CSV)** 로 저장하고 화면에 출력. → 논문 Table 6 같은 최종 결과표.

---

## 🎯 전체 흐름 요약 (한눈에)

```
[데이터 로딩]
  CSV(발열량/유량) + CSV(정답온도) 읽기 → 정규화 → 7채널 시퀀스
        ↓
[모델 생성: HybridModel]
  GRU 인코더 + Branch Net + Trunk Net 6개 + LPTN 물리상수(C, R, A, k, L)
        ↓
[1단계 학습: Adam (2만 회)]
  매 반복마다:
    GRU→Branch→Trunk→내적 = 온도 예측
    물리잔차/초기조건/순차일관성/데이터/teacher 손실 계산
    역전파 → 가중치 업데이트
        ↓
[2단계 학습: L-BFGS (2천 회)]
  정밀 미세조정
        ↓
[검증]
  Direct 예측 + Rollout 예측 → 정답과 비교 → MAPE/sMAPE 계산
  그래프(PNG) + 표(CSV) 저장
```

## 💡 초보자가 특히 기억하면 좋은 5가지

1. **DeepONet = Branch(조건) × Trunk(시간)을 내적**해서 온도 곡선을 만든다. (`output` 함수)
2. **PINN의 핵심은 자동미분**(`torch.autograd.grad`). 신경망 출력을 시간으로 미분해 물리방정식(dT/dt)에 대입한다.
3. **물리 손실 = LPTN 에너지보존식의 잔차를 0에 가깝게** 만드는 것. (10-5의 `res_xxx`)
4. **냉각수 h는 LPM에서 유동 상관식(Re→Nu→h)으로 계산** + 학습 보정계수 α. (8부 `_h_from_LPM...`)
5. **학습은 backward(미분)→step(업데이트)의 반복**. Adam으로 빠르게, L-BFGS로 정밀하게 2단계.

## ⚠️ 코드에서 주의 깊게 본 점

- **10-11부**: 현재 `total = data_loss_s` 한 줄만 활성화되어 있어서, 지금 버전은 **데이터 손실만으로 학습**하도록 임시 설정된 상태입니다(물리/순차/teacher 손실은 계산은 하지만 합산에서 빠짐). 논문대로 물리를 반영하려면 위의 주석 처리된 `total = loss_phy_s + loss_ic_s + ...` 블록을 살려야 합니다.
- 경로(`/home/hye/Documents/...`)가 특정 컴퓨터 기준이라, 다른 환경에서 돌리려면 `DATA_DIR` 등을 본인 경로로 바꿔야 합니다.
