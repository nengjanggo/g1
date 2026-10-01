# g1

Unitree G1에서 VLA policy(unifolm-wla)를 simulation benchmark(IsaacLab-Arena, RoboLab, RoboCasa365)로 평가하는 workspace.
구조는 [docs/BLUEPRINT.md](docs/BLUEPRINT.md) 참고.

## Prerequisites

- Ubuntu 22.04, NVIDIA GPU (RTX 3090 24GB에서 검증). unifolm-wla server만으로 VRAM 약 12GB 사용
- [uv](https://github.com/astral-sh/uv) >= 0.11, git-lfs
- 디스크: checkpoint 약 13GB

## Setup

### 1. Clone

```bash
git clone --recursive <this-repo> g1 && cd g1
# 이미 clone한 경우
git submodule update --init --recursive
```

### 2. g1_eval (이 repo의 client package)

```bash
uv sync
```

### 3. unifolm-wla (policy server)

```bash
cd third_party/unifolm-wla
# --frozen: upstream uv.lock 그대로 설치 (없으면 lock을 재생성해 submodule이 dirty해짐)
# UV_HTTP_TIMEOUT: 네트워크가 느리면 timeout을 늘려야 함
UV_HTTP_TIMEOUT=300 uv sync --frozen
cd ../..
uvx --from huggingface_hub hf download unitreerobotics/UnifoLM-WLA-1.0-Base \
    --local-dir checkpoints/UnifoLM-WLA-1.0-Base
```

`flash-attn`은 선택 사항. 설치하지 않으면 자동으로 `sdpa`로 fallback한다.

## 실행

### Policy server 동작 확인

```bash
# 터미널 1: server 실행 (third_party/unifolm-wla에서)
cd third_party/unifolm-wla
.venv/bin/python -m model_server.action_server_wbc_msgpack_unitree \
    --ckpt_path ../../checkpoints/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors \
    --host 127.0.0.1 --port 8600

# 터미널 2: server log에 'listening on'이 뜬 뒤 dummy obs로 action key/shape 검증 (exit code 0이면 OK)
uv run python scripts/check_policy_server.py --port 8600
```

추론 latency는 server log의 `predict=..ms`로 확인한다.
server 시작 시 출력되는 `TypeError: not enough arguments for format string` logging 에러는 upstream의 무해한 버그이다.

### RTC(Real-Time Chunking) 검증

server 없이 모델을 직접 load한다 (VRAM 약 13GB). 실행 위치는 g1 root:

```bash
# port 단위 self-check (CPU, 수 초)
third_party/unifolm-wla/.venv/bin/python policy_server/rtc.py

# 실제 모델로 검증 (exit code 0이면 OK)
PYTHONPATH=third_party/unifolm-wla:. third_party/unifolm-wla/.venv/bin/python -m policy_server.check_rtc_unifolm \
    --ckpt_path checkpoints/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors
```

문제가 생기면 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) 참고.
