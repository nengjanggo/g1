# TESTING

## Integration test

| 대상 | 검증 내용 | 실행 |
|---|---|---|
| unifolm-wla model_server ↔ [PolicyClient](../src/g1_eval/policy_client.py) | dummy obs(zero image)로 요청했을 때 모든 action key의 shape이 (1, T, D)이고 값이 유한한지 | [check_policy_server.py](../scripts/check_policy_server.py) (README 참고) |

## 검증되지 않은 부분 (known gap)

- 실제 image와 state를 넣었을 때 action이 의미 있는 값인지는 아직 확인하지 않았다. upstream의 `eval_local_episode_wbc_msgpack_server_only.py`로 dataset episode와 비교하는 검증이 남아 있다.
- simulation benchmark 연동(Arena, RoboLab, RoboCasa365)은 아직 구현하지 않았다.
- 실제 hardware 검증은 하지 않았다.
