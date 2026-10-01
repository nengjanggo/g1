# TROUBLESHOOTING

## `uv sync` 중 `operation timed out` (files.pythonhosted.org)

- 증상: unifolm-wla에서 `uv sync`를 실행하면 wheel download가 `Request failed after 3 retries`로 실패한다.
- 원인: 느린 네트워크에서 uv의 기본 HTTP timeout을 넘김
- 해결: `UV_HTTP_TIMEOUT=300 uv sync`

## model_server 시작 시 `TypeError: not enough arguments for format string`

- 증상: server를 시작하면 `Loading model from: %s (backend=%s)` logging traceback이 출력된다.
- 원인: upstream `logging.info` 호출의 인자가 부족한 버그. logging 모듈이 에러를 출력만 하고 실행은 계속된다.
- 해결: server를 실행할 때는 무시해도 된다. `listening on` log가 뜨면 정상이다.
- 단, `ActionServerWBCMsgpack`를 직접 import해서 쓰면 unifolm-wla의 rich logging handler가 이 에러를 예외로 던져 실행이 멈춘다. upstream `main()`처럼 먼저 `logging.basicConfig(force=True)`로 handler를 교체해야 한다.

## `get_action` 시 `No unnorm_key in obs and no default set`

- 원인: UnifoLM-WLA-1.0-Base는 multi-dataset checkpoint라서 normalization stats를 자동으로 고를 수 없다.
- 해결: obs에 `'unnorm_key': 'UnifoLM_G1_Dex1'`(또는 `'UnifoLM_WBT'`)을 넣는다. server를 `--unnorm_key`로 시작해도 된다.
