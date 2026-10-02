'''
unifolm-wla 데이터 규약 ↔ G1-Dex1 sim 값 변환 (numpy/scipy만 사용, Isaac Sim 없이 import 가능).

- EE: unifolm-wla EE 점 = pelvis frame 기준, wrist_yaw_link에서 자기 x축으로 WRIST_TO_EE_X만큼 떨어진 점 (회전 offset 없음).
  WBT Dex1 dataset의 관절값 FK와 기록된 ee_pose_gripper_base를 비교해 추정했다 (frame별 오차 ~2mm).
- gripper: unifolm-wla 값은 Unitree Dex1 명령 단위(5.6 열림 ~ 0 닫힘), sim 관절은 prismatic 위치 [m].
  (출처: unitree_sim_isaaclab tools/data_convert.py, Apache-2.0)

self-check: `third_party/IsaacLab-Arena/.venv/bin/python arena_ext/unifolm_g1_convert.py`
'''

import numpy as np
from scipy.spatial.transform import Rotation

WRIST_TO_EE_X: float = 0.1135
T_WRIST_EE: np.ndarray = np.eye(4)  # (4, 4)
T_WRIST_EE[0, 3] = WRIST_TO_EE_X

# Dex1 손가락 prismatic 관절 위치 [m]
DEX1_OPEN_POS: float = -0.02
DEX1_CLOSE_POS: float = 0.024

# unifolm-wla Dex1 gripper 값에서 완전히 열린 값 (0이면 완전히 닫힘)
DEX1_GRIPPER_OPEN_VALUE: float = 5.6

# unnorm_key별 gripper action 출력의 (완전히 열림, 완전히 닫힘) 값.
# Dex1 통계는 Unitree Dex1 명령 단위, WBT 통계는 gripper를 정규화하지 않아 대략 +1 열림 ~ -1 닫힘으로 나온다
# (WBT Dex1 dataset episode에서 server 예측과 실제 gripper를 비교해 확인)
GRIPPER_ACTION_OPEN_CLOSE: dict[str, tuple[float, float]] = {
    'UnifoLM_G1_Dex1': (DEX1_GRIPPER_OPEN_VALUE, 0.0),
    'UnifoLM_WBT': (1.0, -1.0),
}


def pose7_to_matrix(
    pose7: np.ndarray,
) -> np.ndarray:
    '''pos(3) + quat xyzw(4) pose를 (4, 4) homogeneous transform으로 바꿔 반환한다.'''
    T: np.ndarray = np.eye(4)  # (4, 4)
    T[:3, :3] = Rotation.from_quat(pose7[3:7]).as_matrix()
    T[:3, 3] = pose7[:3]
    return T


def wrist_to_ee_6d(
    T_pelvis_wrist: np.ndarray,
) -> np.ndarray:
    '''
    pelvis 기준 wrist_yaw_link pose (4, 4)에서 unifolm-wla EE state (9,) = xyz + rot6d를 계산해 반환한다.

    rot6d는 회전 행렬의 첫 두 column [R00, R10, R20, R01, R11, R21] (unifolm-wla se3_utils.matrix_to_rot6d와 같음).
    '''
    T_pelvis_ee: np.ndarray = T_pelvis_wrist @ T_WRIST_EE  # (4, 4)
    rotation: np.ndarray = T_pelvis_ee[:3, :3]  # (3, 3)
    return np.concatenate([T_pelvis_ee[:3, 3], rotation[:, 0], rotation[:, 1]]).astype(np.float32)  # (9,)


def ee_rpy_to_wrist_pose(
    ee_xyz_rpy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    '''
    unifolm-wla EE action (6,) = pelvis 기준 xyz + rpy(extrinsic 'xyz')를
    Arena IK 목표인 pelvis 기준 wrist_yaw_link pose로 바꿔 (pos (3,), quat xyzw (4,))로 반환한다.
    '''
    T_pelvis_ee: np.ndarray = np.eye(4)  # (4, 4)
    T_pelvis_ee[:3, :3] = Rotation.from_euler('xyz', ee_xyz_rpy[3:6]).as_matrix()
    T_pelvis_ee[:3, 3] = ee_xyz_rpy[:3]
    T_pelvis_wrist: np.ndarray = T_pelvis_ee @ np.linalg.inv(T_WRIST_EE)  # (4, 4)
    return T_pelvis_wrist[:3, 3], Rotation.from_matrix(T_pelvis_wrist[:3, :3]).as_quat()


def dex1_joint_pos_to_gripper_value(
    joint_pos: float,
) -> float:
    '''Dex1 손가락 prismatic 관절 위치 [m]를 unifolm-wla gripper 값(5.6 열림 ~ 0 닫힘)으로 선형 변환해 반환한다.'''
    return DEX1_GRIPPER_OPEN_VALUE * (DEX1_CLOSE_POS - joint_pos) / (DEX1_CLOSE_POS - DEX1_OPEN_POS)


def gripper_value_to_hand_state(
    gripper_value: float,
    unnorm_key: str,
) -> float:
    '''unnorm_key 단위의 unifolm-wla gripper action 값을 Arena hand_state(0 열림 ~ 1 닫힘)로 선형 변환해 반환한다.'''
    open_value, close_value = GRIPPER_ACTION_OPEN_CLOSE[unnorm_key]
    return float(np.clip((open_value - gripper_value) / (open_value - close_value), 0.0, 1.0))


if __name__ == '__main__':
    # EE 왕복 변환: wrist pose → EE xyz + rot6d → EE xyz + rpy → wrist pose가 원래 값과 같은지
    T_pelvis_wrist: np.ndarray = pose7_to_matrix(np.array([0.3, 0.2, 0.1, *Rotation.from_euler('xyz', [0.3, -0.5, 1.0]).as_quat()]))
    ee_6d: np.ndarray = wrist_to_ee_6d(T_pelvis_wrist)
    ee_rotation: np.ndarray = np.stack([ee_6d[3:6], ee_6d[6:9], np.cross(ee_6d[3:6], ee_6d[6:9])], axis=1)  # (3, 3)
    ee_xyz_rpy: np.ndarray = np.concatenate([ee_6d[:3], Rotation.from_matrix(ee_rotation).as_euler('xyz')])  # (6,)
    wrist_pos, wrist_quat = ee_rpy_to_wrist_pose(ee_xyz_rpy)
    assert np.allclose(wrist_pos, T_pelvis_wrist[:3, 3], atol=1e-5)
    assert np.allclose(Rotation.from_quat(wrist_quat).as_matrix(), T_pelvis_wrist[:3, :3], atol=1e-5)
    # EE 점은 wrist x축 방향으로 WRIST_TO_EE_X 떨어져 있어야 함
    assert np.allclose(ee_6d[:3], T_pelvis_wrist[:3, 3] + WRIST_TO_EE_X * T_pelvis_wrist[:3, 0], atol=1e-5)
    # gripper: 열림 관절 위치 → 5.6 → hand_state 0, 닫힘 관절 위치 → 0 → hand_state 1
    assert np.isclose(dex1_joint_pos_to_gripper_value(DEX1_OPEN_POS), DEX1_GRIPPER_OPEN_VALUE)
    assert np.isclose(dex1_joint_pos_to_gripper_value(DEX1_CLOSE_POS), 0.0)
    assert gripper_value_to_hand_state(DEX1_GRIPPER_OPEN_VALUE, 'UnifoLM_G1_Dex1') == 0.0
    assert gripper_value_to_hand_state(0.0, 'UnifoLM_G1_Dex1') == 1.0
    assert gripper_value_to_hand_state(1.0, 'UnifoLM_WBT') == 0.0
    assert gripper_value_to_hand_state(-1.0, 'UnifoLM_WBT') == 1.0
