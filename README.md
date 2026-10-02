# g1

Unitree G1에서 VLA policy(unifolm-wla)를 simulation benchmark(IsaacLab-Arena, RoboLab, RoboCasa365)로 평가하는 workspace.
구조는 [docs/BLUEPRINT.md](docs/BLUEPRINT.md) 참고.

## Prerequisites

- Ubuntu 22.04, NVIDIA GPU (RTX 3090 24GB에서 검증). unifolm-wla server만으로 VRAM 약 12GB 사용
- [uv](https://github.com/astral-sh/uv) >= 0.11, git-lfs
- 디스크: checkpoint 약 13GB, IsaacLab-Arena venv(Isaac Sim 포함) 약 35GB, Arena scene asset 약 1.2GB

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

### 4. IsaacLab-Arena (Isaac Sim 6 / Isaac Lab 3, Python 3.12)

```bash
cd third_party/IsaacLab-Arena
# OMNI_KIT_ACCEPT_EULA: Isaac Sim EULA 동의 (없으면 설치/실행 중 대화형 prompt에서 멈춤)
OMNI_KIT_ACCEPT_EULA=YES UV_HTTP_TIMEOUT=300 uv sync --frozen
cd ../..
# unifolm-wla server와 통신하는 g1_eval client를 Arena venv에도 설치 (msgpack, websockets는 이미 있음)
uv pip install -e . --python third_party/IsaacLab-Arena/.venv/bin/python --no-deps
```

Isaac Sim wheel이 커서 첫 설치에 30분 이상 걸린다.

### 5. G1-Dex1 robot asset

[unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab)의 asset zip(약 1.2GB)에서 G1 + Dex1 USD(약 51MB)만 `assets/`에 푼다:

```bash
uvx --from huggingface_hub hf download unitreerobotics/unitree_sim_isaaclab_usds assets.zip \
    --repo-type dataset --local-dir /tmp/unitree_usds
unzip -q /tmp/unitree_usds/assets.zip 'assets/robots/g1-29dof_wholebody_dex1/*' -d .
rm -rf /tmp/unitree_usds
```

### 6. Plastic Box Parts 장면 asset

선반/수납함/방/부품 USD를 생성한다. 바닥 텍스처는 Poly Haven의 CC0 텍스처를 받는다:

```bash
mkdir -p assets/textures
curl -L -o assets/textures/laminate_floor_02_diff_1k.jpg \
    https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/laminate_floor_02/laminate_floor_02_diff_1k.jpg
third_party/IsaacLab-Arena/.venv/bin/python arena_ext/plastic_box_parts_scene.py   # assets/generated/plastic_box_parts/
```

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

### IsaacLab-Arena G1 loco-manipulation task 동작 확인

zero action으로 GUI를 띄워 scene과 G1이 뜨는지 확인한다 (third_party/IsaacLab-Arena에서):

```bash
cd third_party/IsaacLab-Arena
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --viz kit --policy_type zero_action --num_steps 3000 --enable_cameras \
    galileo_g1_locomanip_pick_and_place
```

- 첫 실행은 scene asset(약 1.2GB)을 NVIDIA S3에서 `/tmp/https/`로 받느라 20분 이상 걸린다. 이후에는 약 1.5분이다. `/tmp`라서 재부팅하면 다시 받는다.
- GUI 실행 시 약 7 step/s(실시간의 약 0.14배)로 돈다. 대량 평가는 `--viz`를 빼고 headless로 실행한다.
- 결과 report는 `third_party/IsaacLab-Arena/outputs/<timestamp>/index.html`에 생성된다.
- 로그의 `[Error] [omni.rtx.materials]`, `MDLC`, PhysX cooking 경고는 배경 asset의 material/mesh 문제로, 실행에는 영향이 없다.

### Kitchen Bench G1-Dex1 동작 확인

Kitchen Bench pick and place를 G1-Dex1로 띄우고, gripper를 100 step마다 열고 닫는다 (third_party/IsaacLab-Arena에서):

```bash
cd third_party/IsaacLab-Arena
PYTHONPATH=../.. OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --viz kit --policy_type arena_ext.g1_dex1.GripperToggleCheckPolicy --num_steps 1500 --enable_cameras \
    --env_spec ../../configs/arena/kitchen_bench_g1_dex1_pick_and_place.yaml
```

- G1-Dex1 embodiment는 `--policy_type`으로 지정한 `arena_ext` 모듈을 import할 때 등록된다. 그래서 `PYTHONPATH`에 이 repo root가 있어야 한다.
- 첫 실행은 주방 asset을 받느라 오래 걸린다.

### Plastic Box Parts 장면 확인

G1_WBT `Plastic_Box_Parts` task("Move parts from the plastic box to the shelf.")를 본뜬 장면을 띄운다 (third_party/IsaacLab-Arena에서):

```bash
cd third_party/IsaacLab-Arena
PYTHONPATH=../.. OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --viz kit --policy_type arena_ext.g1_dex1.GripperToggleCheckPolicy --num_steps 100000 --enable_cameras \
    --env_spec ../../configs/arena/plastic_box_parts_g1_dex1.yaml
```

- 치수/색은 [plastic_box_parts_scene.py](arena_ext/plastic_box_parts_scene.py) 맨 위 상수를 바꾸고 USD를 다시 생성한다. 배치는 spec YAML의 좌표를 바꾼다. 둘 다 sim을 다시 띄워야 반영된다.

### Kitchen Bench에서 unifolm-wla로 G1-Dex1 제어

server와 sim을 같이 띄우면 VRAM을 약 21GB 쓴다 (server 약 14GB).

```bash
# 터미널 1: server (third_party/unifolm-wla에서). WBT key 요청에 맞는 action mask를 쓰는 진입점
cd third_party/unifolm-wla
PYTHONPATH=.:../.. .venv/bin/python -m policy_server.unitree_server \
    --ckpt_path ../../checkpoints/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors --host 127.0.0.1 --port 8600

# 터미널 2: server log에 'listening on'이 뜬 뒤 sim 실행 (third_party/IsaacLab-Arena에서)
cd third_party/IsaacLab-Arena
PYTHONPATH=../.. OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python isaaclab_arena/evaluation/policy_runner.py \
    --viz kit --policy_type arena_ext.unifolm_wla_policy.UnifolmWlaPolicy --num_episodes 1 --enable_cameras \
    --env_spec ../../configs/arena/kitchen_bench_g1_dex1_pick_and_place.yaml
```

- instruction은 spec YAML의 task `description`이 그대로 들어간다.
- `--unnorm_key`로 dataset 통계를 고른다: `UnifoLM_G1_Dex1`(기본, tabletop 조작) 또는 `UnifoLM_WBT`(걷기 포함 whole-body).
- `--record_camera_video`를 붙이면 머리/손목 카메라 영상이 `outputs/<timestamp>/`에 저장된다.

문제가 생기면 [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) 참고.
