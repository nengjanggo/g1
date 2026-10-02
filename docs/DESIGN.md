# DESIGN

## policy와 simulator를 별도 venv + server-client로 분리

- **결정:** policy(unifolm-wla)는 자체 venv에서 websocket server로 실행한다. 각 benchmark는 자기 venv에서 client로 붙는다.
- **이유:** dependency가 충돌한다. unifolm-wla는 torch 2.8 / transformers 5.5를 쓰고, IsaacLab-Arena는 torch 2.11 / Isaac Sim을 쓰고, RoboCasa는 MuJoCo 계열이다. 한 venv에 설치할 수 없다.
- **trade-off:** 요청마다 직렬화와 통신 비용이 든다. image 3장 기준으로 추론 시간(수백 ms)에 비해 작다.
- client package(`g1_eval`)는 모든 benchmark venv에 설치되므로 numpy, msgpack, websockets만 의존한다.

## RTC를 submodule 수정 없이 주입

- **결정:** unifolm-wla 코드를 수정하지 않는다. 대신 [unifolm_rtc.py](../policy_server/unifolm_rtc.py)의 `enable_rtc`가 실행 중에 model instance의 `predict_action` 두 개를 교체한다.
  - framework `predict_action`: `@torch.inference_mode`를 `no_grad`로 바꾼다. inference mode 안에서는 `enable_grad`로도 RTC의 VJP를 계산할 수 없다.
  - action head `predict_action`: sampling loop를 RTC 버전으로 바꾼다.
- **이유:** submodule은 upstream commit에 고정돼 있고 fork가 없다. 직접 수정하면 submodule이 dirty해지고, 그 변경을 push할 곳이 없어 재현이 안 된다.
- **검토한 대안:**
  - fork 후 submodule URL 변경: upstream을 따라가기가 번거롭다.
  - patch 파일 관리: 적용 단계가 하나 늘어난다.
- **trade-off:** upstream의 sampling loop body를 복제했다 ([make_velocity_fn](../policy_server/unifolm_rtc.py)). upstream이 loop를 바꾸면 따라 고쳐야 한다. `max_guidance_weight=0` 등가성 check가 이런 drift를 잡는다 ([TESTING.md](TESTING.md)).

## RTC는 원본 구현을 그대로 port

- 원본(Physical Intelligence, JAX)의 알고리즘과 상수를 그대로 쓴다. 기본값도 원본과 같다: schedule `exp`, `max_guidance_weight=5.0`.
- RTC는 soft guidance다. 원본 eval도 앞쪽 `inference_delay` step은 **이전 chunk에서 실행**하고, 새 chunk는 그 이후부터 쓴다. 새 chunk의 prefix가 이전 chunk와 정확히 같을 필요는 없다.
- `max_guidance_weight`를 크게 잡으면(예: 100) 4-step Euler에서 overshoot해서 발산하는 것을 확인했다. 그래서 상한은 원본 기본값을 유지한다.

## G1-Dex1을 Arena WBC에 붙이는 방법

- **결정:** Arena의 G1(Dex3) 대신 unitree_sim_isaaclab의 G1 + Dex1 USD를 쓰고, Arena WBC가 가정하는 43-dof(Dex3) 관절 배열과의 차이는 [g1_dex1.py](../arena_ext/g1_dex1.py)에서 메운다.
  - 관측: sim에 없는 Dex3 hand 관절 10개를 0으로 채워 43개로 맞춘 robot data proxy를 WBC에 넘긴다.
  - 출력: WBC 출력 중 sim에 있는 관절만 쓰고, Dex1 관절 4개는 Dex3 hand slot 4개에 배정한 뒤 `hand_state`로 덮어쓴다.
- **이유:** unifolm-wla는 Dex1, Inspire, BrainCo로만 학습됐고 Dex3 데이터가 없다. 손목 카메라에 보이는 gripper 모양까지 학습 데이터와 맞추려면 Dex1 모델이 필요하다.
- **검토한 대안:** Arena의 Dex3 G1에 손목 카메라만 추가하고 gripper 값을 Dex3 open/close로 매핑. WBC를 건드리지 않아 단순하지만, 모델이 본 적 없는 손이 된다.
- **trade-off:** Arena 내부 함수(`prepare_observations`, `postprocess_actions`)를 module 단위로 교체한다. Arena가 이 함수들의 signature나 관절 가정을 바꾸면 깨진다. 관절 수가 어긋나면 Arena의 assert에서 바로 실패하므로 조용히 틀릴 가능성은 낮다.

## unifolm-wla ↔ Arena G1 변환

[unifolm_g1_convert.py](../arena_ext/unifolm_g1_convert.py), [unifolm_wla_policy.py](../arena_ext/unifolm_wla_policy.py)의 변환 규약과 근거.

- **EE pose:** unifolm-wla의 `ee_pose_gripper_base`는 pelvis frame 기준이고, EE 점은 `wrist_yaw_link`에서 자기 x축으로 약 0.11m 떨어진 점이다(회전 offset 없음). WBT Dex1 dataset의 관절값 FK와 기록된 EE pose를 비교해 정했다(frame별 오차 약 2mm, torso frame으로 가정하면 오차가 더 크다). Arena PINK IK 목표는 pelvis frame의 `wrist_yaw_link` pose라서 이 offset만 붙였다 뗀다.
  - 이 dataset의 팔 관절 이름은 `wrist_yaw, wrist_roll, wrist_pitch` 순서로 적혀 있지만 값은 G1 물리 순서(`wrist_roll, wrist_pitch, wrist_yaw`)다.
- **하체:** 학습 데이터의 다리 관절은 Unitree RL controller가 `base_command`를 실행한 결과다. 다른 controller(Arena WBC)에서 다리 관절을 직접 따라가면 균형을 잃을 수 있어 `base_command`만 WBC에 넘긴다. `base_command`의 3번째 값은 dataset 이름(`angle_z`)과 달리 yaw rate다(측정된 base yaw rate와 상관 0.77, yaw 각도와는 0.02).
- **gripper:** obs의 gripper state는 Dex1 명령 단위(5.6 열림 ~ 0 닫힘)로 보낸다. action 출력 단위는 `unnorm_key`의 정규화 통계에 따라 다르다(Dex1은 같은 단위, WBT는 gripper를 정규화하지 않아 약 ±1).
- **머리 카메라:** Arena G1 머리 카메라(수평 화각 약 70°)는 실제 G1 데이터보다 좁다. unitree_sim_isaaclab의 G1 설정(`d435_link`, 약 48° 아래, 수평 화각 약 105°)으로 바꿨다.

## WBT 요청의 action mask

- **결정:** [unitree_server.py](../policy_server/unitree_server.py)에서 `UnifoLM_WBT` 요청에만 upstream mask에 base pose 자리 [35:41]을 더한 mask를 쓴다.
- **이유:** action mask는 action head의 조건 입력이다. WBT 데이터는 학습 때 base pose 자리가 켜져 있었는데 upstream server는 항상 Dex1 기준 mask(base pose 꺼짐)를 쓴다. 이 mask로는 WBT dataset의 걷는 구간(실제 vx 약 0.11m/s)에서도 예측 vx가 0 근처였고, base pose를 켜면 같은 구간에서 전진을 예측했다(서 있는 구간은 0 근처 유지).
- **trade-off:** WBT 학습 설정에는 gripper key가 없어서 학습 때 WBT의 gripper 자리는 꺼져 있었을 수 있다. sim에서 gripper를 제어하려고 켜 두었으므로 WBT 조건과 완전히 같지는 않다.
- **검토한 대안:** submodule의 server mask 수정. submodule은 upstream commit에 고정하고 수정하지 않는다는 원칙([RTC 주입](#rtc를-submodule-수정-없이-주입)과 같은 이유)으로 채택하지 않았다.
