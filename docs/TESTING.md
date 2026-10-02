# TESTING

## Integration test

| 대상 | 검증 내용 | 실행 |
|---|---|---|
| unifolm-wla model_server ↔ [PolicyClient](../src/g1_eval/policy_client.py) | dummy obs(zero image)로 요청했을 때 모든 action key의 shape이 (1, T, D)이고 값이 유한한지 | [check_policy_server.py](../scripts/check_policy_server.py) (README 참고) |
| RTC + 실제 unifolm-wla 모델 | 1) `max_guidance_weight=0`이면 같은 seed에서 원본 sampling과 같은지. 2) 모델 자신의 다른 sample을 prev chunk로 주면 frozen prefix가 그쪽으로 끌려오는지 | [check_rtc_unifolm.py](../policy_server/check_rtc_unifolm.py) (README 참고) |

## Simulation test (수동, GUI 확인)

| 대상 | 검증 내용 | 결과 |
|---|---|---|
| Arena G1-Dex1 WBC 걷기 | 모델 없이 `navigate_cmd` vx 0.2m/s를 5초 넣었을 때 따라가는지 | 시작 후 약 1초 지연 뒤 약 0.2m/s로 전진, 정지 명령에 멈춤 (일회성 확인 script는 repo에 남기지 않음) |
| Kitchen Bench + unifolm-wla ([unifolm_wla_policy.py](../arena_ext/unifolm_wla_policy.py)) | server와 sim이 끝까지 연결되어 도는지 | 동작함. 팔은 몇 cm만 움직이고, 걷기 프롬프트에도 전진 명령을 거의 내지 않음 |

## Unit test

| 대상 | 검증 내용 | 실행 |
|---|---|---|
| [unifolm_g1_convert.py](../arena_ext/unifolm_g1_convert.py) | EE pose 왕복 변환, EE offset 방향, key별 gripper 단위 변환 | `third_party/IsaacLab-Arena/.venv/bin/python arena_ext/unifolm_g1_convert.py` |
| [rtc.py](../policy_server/rtc.py) port | prefix weight 예시값, guidance 0일 때 Euler와 같은지, toy flow에서 prefix를 따라가는지 | `third_party/unifolm-wla/.venv/bin/python policy_server/rtc.py` |

## 검증되지 않은 부분 (known gap)

- dataset episode 대비 open-loop 검증은 WBT Dex1 dataset(`G1_WBT_Dex1_Put_Clothes_into_Washing_Machine`)에서 일회성 script로만 했다(repo에 남기지 않음): EE 예측 오차가 정지 기준선보다 작고, [unitree_server.py](../policy_server/unitree_server.py)의 WBT mask에서 걷는 구간의 전진을 예측했다. 이 episode는 학습에 쓰였을 가능성이 높아 일반화 검증은 아니다. upstream의 `eval_local_episode_wbc_msgpack_server_only.py`는 전처리된 로컬 dataset과 현재 repo에 없는 import 경로를 요구해 쓰지 않았다.
- RTC는 sampling 단위로만 검증했다. 아직 하지 않은 것: server protocol 통합(prev chunk를 현재 state 기준으로 다시 표현, inference_delay 측정)과 closed-loop에서 실제로 부드러워지는지 확인.
- unifolm-wla 연동은 Arena Kitchen Bench pick and place 하나에서만 수동으로 돌려봤다. success rate 평가, RoboLab, RoboCasa365 연동은 하지 않았다.
- G1-Dex1 embodiment([g1_dex1.py](../arena_ext/g1_dex1.py))는 Kitchen Bench에서 GUI로 눈으로만 확인했다 (README의 "Kitchen Bench G1-Dex1 동작 확인"): G1-Dex1이 서 있고(`base_height_cmd` 0.75m) gripper가 열리고 닫히는지, 머리 카메라에 조리대 위 물체가 보이는지. 손목 카메라 영상 내용은 확인하지 않았다.
- 팔 목표 추종: unifolm-wla 실행 중 손목 목표 대비 실제 손목 z가 3~4cm 낮게 처지는 것을 기록에서 확인했다. 원인(팔 gain, 중력)은 확인하지 않았다. PINK IK의 robot model은 Dex3 기준이라 Dex1 손 끝 위치와의 차이도 확인하지 않았다.
- 실제 hardware 검증은 하지 않았다.
