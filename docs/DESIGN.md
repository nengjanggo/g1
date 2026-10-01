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
