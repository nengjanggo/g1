# BLUEPRINT

## 전체 구조

policy(unifolm-wla)와 simulator(Isaac Sim, MuJoCo)는 dependency가 충돌하므로 각자 독립된 venv에서 실행하고,
websocket + msgpack server-client로 통신한다.

```
[unifolm-wla venv] model_server  <-- websocket/msgpack -->  [benchmark venv] g1_eval client + simulator
```

| 경로 | 역할 |
|---|---|
| [src/g1_eval/](../src/g1_eval/) | 직접 작성하는 client package. 모든 benchmark venv에 설치되므로 numpy, msgpack, websockets만 의존 |
| [policy_server/](../policy_server/) | unifolm-wla venv에서 실행되는 server 쪽 확장 (RTC 등). submodule을 import해서 쓰고, submodule 파일은 수정하지 않음 |
| [arena_ext/](../arena_ext/) | IsaacLab-Arena venv에서 실행되는 Arena 확장 (G1-Dex1 embodiment 등). submodule을 import해서 쓰고, submodule 파일은 수정하지 않음 |
| [configs/arena/](../configs/arena/) | Arena environment graph spec YAML (`--env_spec`) |
| [scripts/](../scripts/) | 실행/검증 entry point |
| [third_party/](../third_party/) | 외부 repo (git submodule, commit 고정). 각자 자체 `.venv` 사용 |
| `checkpoints/`, `data/`, `outputs/`, `assets/` | 대용량/generated/외부 asset 파일 (gitignore). `assets/`는 README의 G1-Dex1 asset 추출로 만든다 |

## g1_eval

### [policy_client.py](../src/g1_eval/policy_client.py)

`PolicyClient(host, port)`: unifolm-wla model_server에 연결하는 client.
- 연결하면 server가 metadata(`data_keys`, `action_chunk_size` 등)를 먼저 보내고, 이를 `self.metadata`에 저장한다.
- `get_action(obs)`: `{'type': 'get_action', 'obs': obs}`를 보내고 action dict를 반환한다.
- codec은 model_server의 [msgpack_numpy.py](../third_party/unifolm-wla/model_server/tools/msgpack_numpy.py)와 같은 포맷이다. submodule을 import하지 않도록 client 쪽에 따로 두었다.

## policy_server

unifolm-wla venv와 `PYTHONPATH=third_party/unifolm-wla:.`로 실행한다.

### [rtc.py](../policy_server/rtc.py)

Real-Time Chunking(RTC) guided flow sampling. Physical Intelligence의 공식 JAX 구현을 PyTorch로 port했다 (출처와 license는 파일 header에 표기).
- `realtime_action(velocity_fn, noise, ...)`: 모델과 무관한 generic sampler. `velocity_fn(x_t, t) -> v_t`만 받는다.
- flow time convention: t=0이 noise, t=1이 data (unifolm-wla와 같음)

### [unifolm_rtc.py](../policy_server/unifolm_rtc.py)

`enable_rtc(model, prev_action_chunk, ...)`: unifolm-wla framework model의 `predict_action`이 RTC sampling을 쓰도록 **instance 단위로** 교체한다.
- `prev_action_chunk`가 요청마다 바뀌므로 매 추론 전에 다시 호출한다.
- `prev_action_chunk`는 현재 state 기준의 **normalized unified action** (B, H, 54)이어야 한다. EE와 base pose action이 현재 state에 대한 상대값이라, 이전 chunk를 현재 state 기준으로 다시 표현해야 한다.
- 이유와 대안은 [DESIGN.md](DESIGN.md#rtc를-submodule-수정-없이-주입) 참고.

### [unitree_server.py](../policy_server/unitree_server.py)

upstream model_server 실행 진입점. 인자와 protocol은 upstream과 같고, `unnorm_key`가 `UnifoLM_WBT`인 요청에만 base pose 자리를 켠 action mask를 넘긴다. 이유는 [DESIGN.md](DESIGN.md#wbt-요청의-action-mask) 참고.

## unifolm-wla model_server contract

server: [action_server_wbc_msgpack_unitree.py](../third_party/unifolm-wla/model_server/action_server_wbc_msgpack_unitree.py)

- **Dex1(2-finger gripper) 전용.** WBT(full-body) checkpoint의 dexterous hand(`fig6d`) 및 base pose action은 protocol에 없다. 지원하려면 server의 `_DATA_KEYS`, `_build_state_unnorm`, `_encode_action`을 확장해야 한다.
- obs:
  - image 3개(`cam_left_high`, `cam_left_wrist`, `cam_right_wrist`): 각각 (H, W, 3) uint8 BGR
  - `left/right_ee_6d`: (9,) xyz + rot6d. robot base frame 기준 (x 전방, y 좌측, z 상방)
  - `left/right_gripper`: (1,)
  - `lower_body`: (15,) = left_leg(6) + right_leg(6) + waist(3)
  - `instruction`(선택)
  - `unnorm_key`: 필수. Base checkpoint는 multi-dataset이라 `UnifoLM_G1_Dex1` 또는 `UnifoLM_WBT` 중 하나를 지정해야 한다.
- action: 모든 값은 (1, T, D)이고 **unnormalized absolute** 값이다. T = `action_chunk_size` (기본 30, 30 FPS 기준 1초)
  - `left/right_ee_rpy` (D=6): xyz + rpy 절대 EE pose
  - `left/right_gripper` (D=1)
  - `lower_body` (D=15)
  - `base_command` (D=4): vx, vy, vw, height
  - `pivot` (D=7)
- action mask는 모델의 조건 입력이다. upstream server는 항상 Dex1 기준 mask를 쓰므로 WBT key로 요청할 때는 [unitree_server.py](../policy_server/unitree_server.py)를 쓴다.
- `left/right_gripper` action 단위는 `unnorm_key`마다 다르다 (Dex1: 5.6 열림 ~ 0 닫힘, WBT: 약 +1 열림 ~ -1 닫힘). obs의 gripper state는 두 key 모두 Dex1 명령 단위(5.6 열림 ~ 0 닫힘)로 보낸다.
- 모델 내부의 통합 action/state 공간은 54-D / 60-D이다: [robot_action_state_processing_en.md](../third_party/unifolm-wla/docs/robot_action_state_processing_en.md)

## arena_ext

IsaacLab-Arena venv와 `PYTHONPATH=<repo root>`로 실행한다. Arena 모듈은 sim app이 뜬 뒤에만 import할 수 있으므로,
`policy_runner.py --policy_type arena_ext.<module>.<Policy>`로 policy를 불러올 때 같은 모듈의 embodiment도 함께 등록된다.

### [g1_dex1.py](../arena_ext/g1_dex1.py)

`g1_dex1_wbc_pink` embodiment: Arena의 `g1_wbc_pink`(G1 + Dex3)에서 robot USD를 G1 + Dex1으로, 카메라를 unitree_sim_isaaclab의 G1 머리 카메라 + Dex1 손목 2개로 바꾼 것.
- action: `g1_wbc_pink`와 같은 23-D layout. `left/right_hand_state`는 [0, 1]로 잘린 뒤 Dex1 open/close 위치 사이로 선형 보간된다.
- observation 카메라: `robot_head_cam_rgb`, `left_wrist_cam_rgb`, `right_wrist_cam_rgb` (각 480x640x3).
- Arena WBC는 43-dof(Dex3) 관절 배열을 가정한다. 그래서 import 시 Arena WBC의 관측/출력 변환 함수를 Dex1용으로 교체한다. 이유는 [DESIGN.md](DESIGN.md#g1-dex1을-arena-wbc에-붙이는-방법) 참고.
- G1 WBC는 50Hz 제어를 가정하므로 spec YAML에 `env_cfg_override`(dt 0.005, decimation 4)가 있어야 한다 (예: [kitchen_bench_g1_dex1_pick_and_place.yaml](../configs/arena/kitchen_bench_g1_dex1_pick_and_place.yaml)).

### [unifolm_wla_policy.py](../arena_ext/unifolm_wla_policy.py)

`UnifolmWlaPolicy` (`--policy_type arena_ext.unifolm_wla_policy.UnifolmWlaPolicy`): G1-Dex1 Arena 관측을 unifolm-wla server obs로 바꿔 보내고, 받은 chunk(30 FPS)를 `g1_dex1_wbc_pink`의 23-D action으로 바꿔 실행한다.
- `num_envs=1` 전용. `replan_steps` sim step마다 chunk를 새로 받는다 (sim은 추론 동안 멈추므로 동기 실행).
- 실행에 쓰는 출력: 양손 EE pose, gripper, `base_command`(속도 3 + 골반 높이), waist(torso 자세 명령으로 근사). 다리 관절 출력은 쓰지 않는다.

### [unifolm_g1_convert.py](../arena_ext/unifolm_g1_convert.py)

unifolm-wla 데이터 규약과 G1-Dex1 sim 값 사이의 변환 (numpy/scipy만 사용, sim 없이 import 가능). EE pose(pelvis frame, gripper 점 ↔ `wrist_yaw_link`)와 gripper 단위(Dex1 관절 위치 ↔ unifolm-wla 값 ↔ Arena `hand_state`)를 다룬다. 근거는 [DESIGN.md](DESIGN.md#unifolm-wla--arena-g1-변환).

### [plastic_box_parts_scene.py](../arena_ext/plastic_box_parts_scene.py)

G1_WBT `Plastic_Box_Parts` task 장면 asset. 실행 방식이 둘이다.
- `__main__`: sim 없이 pxr로 방(바닥 + 벽 + 천장 조명), 선반, 수납함, 부품 USD를 `assets/generated/plastic_box_parts/`에 생성한다. Arena와 무관한 일반 USD라 다른 sim에서도 쓸 수 있다.
- import(sim app이 뜬 뒤): 생성된 USD와 procedural 골판지 상자, 밝은 dome light를 Arena asset으로 등록한다(`plastic_box_parts_room`, `boltless_shelf`, `storage_tote`, `cardboard_insert`, `part_tray`, `bright_dome_light`). [g1_dex1.py](../arena_ext/g1_dex1.py)가 import한다.
- 배치는 [plastic_box_parts_g1_dex1.yaml](../configs/arena/plastic_box_parts_g1_dex1.yaml)에서 좌표로 정한다. 부품은 실제 asset을 찾기 전까지 쓰는 위가 파인 쟁반 모양이다.
