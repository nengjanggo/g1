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
| [scripts/](../scripts/) | 실행/검증 entry point |
| [third_party/](../third_party/) | 외부 repo (git submodule, commit 고정). 각자 자체 `.venv` 사용 |
| `checkpoints/`, `data/`, `outputs/` | 대용량/generated 파일 (gitignore) |

## g1_eval

### [policy_client.py](../src/g1_eval/policy_client.py)

`PolicyClient(host, port)`: unifolm-wla model_server에 연결하는 client.
- 연결하면 server가 metadata(`data_keys`, `action_chunk_size` 등)를 먼저 보내고, 이를 `self.metadata`에 저장한다.
- `get_action(obs)`: `{'type': 'get_action', 'obs': obs}`를 보내고 action dict를 반환한다.
- codec은 model_server의 [msgpack_numpy.py](../third_party/unifolm-wla/model_server/tools/msgpack_numpy.py)와 같은 포맷이다. submodule을 import하지 않도록 client 쪽에 따로 두었다.

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
- 모델 내부의 통합 action/state 공간은 54-D / 60-D이다: [robot_action_state_processing_en.md](../third_party/unifolm-wla/docs/robot_action_state_processing_en.md)
