'''
실행 중인 unifolm-wla model_server에 dummy obs를 보내 action chunk의 key/shape를 검증한다.
추론 latency는 server log의 predict=..ms 로 확인한다.

사용법:
    uv run python scripts/check_policy_server.py --host 127.0.0.1 --port 8600
'''
import argparse

import numpy as np

from g1_eval.policy_client import PolicyClient


def make_dummy_obs(
    image_height: int,
    image_width: int,
) -> dict[str, np.ndarray | str]:
    '''
    unifolm-wla Dex1 server protocol의 observation.* key를 모두 채운 dummy obs dict를 반환한다.
    EE pose는 identity rotation, 나머지 state는 0으로 채운다.
    '''
    # 카메라 3개 dummy image: (image_height, image_width, 3) uint8 BGR
    image: np.ndarray = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    # EE pose: xyz + rot6d(identity의 첫 두 column) → (9,)
    left_ee_6d: np.ndarray = np.array([0.3, 0.2, 0.1, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    right_ee_6d: np.ndarray = np.array([0.3, -0.2, 0.1, 1, 0, 0, 0, 1, 0], dtype=np.float32)
    return {
        'observation.images.cam_left_high': image,
        'observation.images.cam_left_wrist': image,
        'observation.images.cam_right_wrist': image,
        'observation.state.left_ee_6d': left_ee_6d,
        'observation.state.right_ee_6d': right_ee_6d,
        'observation.state.left_gripper': np.zeros(1, dtype=np.float32),   # (1,)
        'observation.state.right_gripper': np.zeros(1, dtype=np.float32),  # (1,)
        'observation.state.lower_body': np.zeros(15, dtype=np.float32),    # (15,) left_leg(6) + right_leg(6) + waist(3)
        'instruction': 'pick up the cup',
        # multi-dataset checkpoint의 normalization stats 선택 (UnifoLM_G1_Dex1 | UnifoLM_WBT)
        'unnorm_key': 'UnifoLM_G1_Dex1',
    }


def main(
) -> None:
    '''dummy obs로 get_action을 n_requests번 호출해 action key/shape를 검증한다.'''
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8600)
    parser.add_argument('--n_requests', type=int, default=5)
    args: argparse.Namespace = parser.parse_args()

    # server 연결 및 metadata 확인
    client: PolicyClient = PolicyClient(host=args.host, port=args.port)
    action_chunk_size: int = client.metadata['action_chunk_size']
    obs: dict[str, np.ndarray | str] = make_dummy_obs(image_height=480, image_width=640)

    # 기대하는 action key별 마지막 차원 D
    expected_dims: dict[str, int] = {
        'action.left_ee_rpy': 6,
        'action.right_ee_rpy': 6,
        'action.left_gripper': 1,
        'action.right_gripper': 1,
        'action.lower_body': 15,
        'action.base_command': 4,
        'action.pivot': 7,
    }

    for _ in range(args.n_requests):
        # action chunk 요청
        action: dict[str, np.ndarray] = client.get_action(obs=obs)
        # 각 action: (1, action_chunk_size, D), 유한값인지 검증
        for key, dim in expected_dims.items():
            assert action[key].shape == (1, action_chunk_size, dim), (key, action[key].shape)
            assert np.isfinite(action[key]).all(), key
    client.close()


if __name__ == '__main__':
    main()
