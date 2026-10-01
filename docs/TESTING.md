# TESTING

## Integration test

| 대상 | 검증 내용 | 실행 |
|---|---|---|
| unifolm-wla model_server ↔ [PolicyClient](../src/g1_eval/policy_client.py) | dummy obs(zero image)로 요청했을 때 모든 action key의 shape이 (1, T, D)이고 값이 유한한지 | [check_policy_server.py](../scripts/check_policy_server.py) (README 참고) |
| RTC + 실제 unifolm-wla 모델 | 1) `max_guidance_weight=0`이면 같은 seed에서 원본 sampling과 같은지. 2) 모델 자신의 다른 sample을 prev chunk로 주면 frozen prefix가 그쪽으로 끌려오는지 | [check_rtc_unifolm.py](../policy_server/check_rtc_unifolm.py) (README 참고) |

## Unit test

| 대상 | 검증 내용 | 실행 |
|---|---|---|
| [rtc.py](../policy_server/rtc.py) port | prefix weight 예시값, guidance 0일 때 Euler와 같은지, toy flow에서 prefix를 따라가는지 | `third_party/unifolm-wla/.venv/bin/python policy_server/rtc.py` |

## 검증되지 않은 부분 (known gap)

- 실제 image와 state를 넣었을 때 action이 의미 있는 값인지는 아직 확인하지 않았다. upstream의 `eval_local_episode_wbc_msgpack_server_only.py`로 dataset episode와 비교하는 검증이 남아 있다.
- RTC는 sampling 단위로만 검증했다. 아직 하지 않은 것: server protocol 통합(prev chunk를 현재 state 기준으로 다시 표현, inference_delay 측정)과 closed-loop에서 실제로 부드러워지는지 확인.
- simulation benchmark와 unifolm-wla 연동(Arena, RoboLab, RoboCasa365)은 아직 구현하지 않았다. Arena는 설치 후 zero action으로 환경 단독 실행만 수동으로 한 번 확인했다 (README 참고).
- G1-Dex1 embodiment([g1_dex1.py](../arena_ext/g1_dex1.py))는 Kitchen Bench에서 GUI로 눈으로만 확인했다 (README의 "Kitchen Bench G1-Dex1 동작 확인"): scene과 G1-Dex1이 뜨고 gripper가 열리고 닫히는지. 이 확인용 policy에서 G1은 넘어진다. `base_height_cmd`까지 0으로 보내기 때문으로 추정하지만 확인하지 않았다. 아직 하지 않은 것: 정상 명령에서 서 있는지, 손목 카메라 이미지 내용 확인, 걷기/팔 이동 명령에서 WBC 동작 확인. PINK IK의 robot model은 Dex3 기준이라 Dex1 손 끝 위치와의 차이도 확인하지 않았다.
- 실제 hardware 검증은 하지 않았다.
