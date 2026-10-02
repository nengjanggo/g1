'''
IsaacLab-Arena에서 G1-Dex1(arena_ext/g1_dex1.py)을 unifolm-wla model_server로 제어하는 policy.

Arena 관측(카메라 3개, robot state)을 unifolm-wla server의 obs 형식으로 바꿔 보내고,
받은 action chunk(30 FPS, 절대값)를 Arena `g1_dex1_wbc_pink`의 23-D action으로 바꿔 50Hz로 실행한다.
좌표/단위 변환은 arena_ext/unifolm_g1_convert.py 참고.

사용: `--policy_type arena_ext.unifolm_wla_policy.UnifolmWlaPolicy` (g1_dex1 embodiment도 이 import로 등록된다)
'''

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from gymnasium.spaces.dict import Dict as GymSpacesDict
from isaaclab.assets.articulation import ArticulationData

import arena_ext.g1_dex1  # noqa: F401  g1_dex1_wbc_pink embodiment 등록
from arena_ext.unifolm_g1_convert import (
    dex1_joint_pos_to_gripper_value,
    ee_rpy_to_wrist_pose,
    gripper_value_to_hand_state,
    pose7_to_matrix,
    wrist_to_ee_6d,
)
from g1_eval.policy_client import PolicyClient
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_HAND_STATE_IDX,
    LEFT_WRIST_POS_START_IDX,
    LEFT_WRIST_QUAT_START_IDX,
    NAVIGATE_CMD_START_IDX,
    RIGHT_HAND_STATE_IDX,
    RIGHT_WRIST_POS_START_IDX,
    RIGHT_WRIST_QUAT_START_IDX,
    TORSO_ORIENTATION_RPY_CMD_START_IDX,
)

# unifolm-wla lower_body 순서: left_leg(6) + right_leg(6) + waist(yaw, roll, pitch)
LOWER_BODY_JOINT_NAMES: list[str] = [
    f'{side}_{joint}_joint'
    for side in ['left', 'right']
    for joint in ['hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll']
] + ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']

# Arena camera obs 이름 → unifolm-wla image key
CAMERA_TO_IMAGE_KEY: dict[str, str] = {
    'robot_head_cam_rgb': 'observation.images.cam_left_high',
    'left_wrist_cam_rgb': 'observation.images.cam_left_wrist',
    'right_wrist_cam_rgb': 'observation.images.cam_right_wrist',
}


def body_pose_in_pelvis(
    robot_data: ArticulationData,
    body_name: str,
) -> np.ndarray:
    '''env 0에서 body_name link의 pose를 pelvis frame 기준 (4, 4) homogeneous transform으로 반환한다.'''
    # world 기준 link pose (pos 3 + quat xyzw 4)
    body_state_w: torch.Tensor = wp.to_torch(robot_data.body_link_state_w)[0]  # (num_bodies, 13)
    pelvis_pose_w: np.ndarray = body_state_w[robot_data.body_names.index('pelvis'), :7].cpu().numpy()  # (7,)
    body_pose_w: np.ndarray = body_state_w[robot_data.body_names.index(body_name), :7].cpu().numpy()  # (7,)
    T_world_pelvis: np.ndarray = pose7_to_matrix(pelvis_pose_w)  # (4, 4)
    T_world_body: np.ndarray = pose7_to_matrix(body_pose_w)  # (4, 4)
    return np.linalg.inv(T_world_pelvis) @ T_world_body  # (4, 4)












@dataclass
class UnifolmWlaPolicyCfg(PolicyCfg):
    '''unifolm-wla model_server 연결과 chunk 실행 설정.'''

    # chunk를 몇 sim step마다 새로 받을지. 50Hz에서 50 step = chunk 하나(30 frame, 1초)를 끝까지 실행
    replan_steps: int = 50

    policy_host: str = '127.0.0.1'

    policy_port: int = 8600

    # Base checkpoint의 dataset 통계 key. G1-Dex1이므로 Dex1 통계 사용
    unnorm_key: str = 'UnifoLM_G1_Dex1'

    # unifolm-wla action chunk의 frame rate (학습 데이터 30 FPS)
    policy_fps: float = 30.0


@register_policy
class UnifolmWlaPolicy(PolicyBase[UnifolmWlaPolicyCfg]):
    '''unifolm-wla server에서 action chunk를 받아 G1-Dex1 Arena action으로 실행하는 policy (num_envs=1 전용).'''

    name = 'unifolm_wla'

    def __init__(
        self,
        config: UnifolmWlaPolicyCfg,
    ):
        '''server에 연결하고 chunk 상태를 초기화한다.'''
        super().__init__(config)
        self.client: PolicyClient = PolicyClient(config.policy_host, config.policy_port)
        self.action_chunk: dict[str, np.ndarray] | None = None
        self.steps_since_chunk: int = 0

    def reset(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        '''episode가 바뀌면 받아 둔 chunk를 버린다.'''
        self.action_chunk = None
        self.steps_since_chunk = 0

    def close(
        self,
    ) -> None:
        '''server 연결을 닫는다.'''
        self.client.close()

    def build_obs(
        self,
        env: gym.Env,
        observation: GymSpacesDict,
    ) -> dict:
        '''Arena 관측에서 unifolm-wla server obs dict(카메라 3개 BGR, EE 6d, gripper, lower_body, instruction)를 만든다.'''
        robot_data: ArticulationData = env.unwrapped.scene['robot'].data
        joint_pos: np.ndarray = wp.to_torch(robot_data.joint_pos)[0].cpu().numpy()  # (num_joints,)
        joint_names: list[str] = list(robot_data.joint_names)

        # 카메라 이미지: Arena RGB (H, W, 3) → server가 기대하는 uint8 BGR
        obs: dict = {
            image_key: observation['camera_obs'][camera_name][0].clamp(0, 255).to(torch.uint8).cpu().numpy()[..., ::-1].copy()
            for camera_name, image_key in CAMERA_TO_IMAGE_KEY.items()
        }
        # 양손 EE state: pelvis 기준 wrist_yaw_link pose에서 EE 점의 xyz + rot6d
        obs['observation.state.left_ee_6d'] = wrist_to_ee_6d(body_pose_in_pelvis(robot_data, 'left_wrist_yaw_link'))
        obs['observation.state.right_ee_6d'] = wrist_to_ee_6d(body_pose_in_pelvis(robot_data, 'right_wrist_yaw_link'))
        # gripper state: 손가락 1번 관절 위치를 unifolm-wla gripper 단위로 변환
        for side in ['left', 'right']:
            finger_pos: float = float(joint_pos[joint_names.index(f'{side}_hand_Joint1_1')])
            obs[f'observation.state.{side}_gripper'] = np.array([dex1_joint_pos_to_gripper_value(finger_pos)], dtype=np.float32)
        # 다리 12개 + waist 3개 관절 위치
        obs['observation.state.lower_body'] = joint_pos[[joint_names.index(n) for n in LOWER_BODY_JOINT_NAMES]].astype(np.float32)
        obs['instruction'] = self.task_description
        obs['unnorm_key'] = self.config.unnorm_key
        return obs

    def get_action(
        self,
        env: gym.Env,
        observation: GymSpacesDict,
    ) -> torch.Tensor:
        '''
        replan_steps마다 server에서 chunk를 새로 받고, 현재 시각에 해당하는 chunk frame을 Arena action으로 바꿔 반환한다.

        반환 shape: (num_envs, action_dim)
        '''
        # chunk가 없거나 replan 시점이면 새 chunk 요청
        if self.action_chunk is None or self.steps_since_chunk >= self.config.replan_steps:
            self.action_chunk = self.client.get_action(self.build_obs(env, observation))
            self.steps_since_chunk = 0
        # sim 경과 시간을 chunk frame index로 변환 (50Hz sim step → 30 FPS frame)
        chunk_len: int = self.action_chunk['action.base_command'].shape[1]
        frame_idx: int = min(int(self.steps_since_chunk * env.unwrapped.step_dt * self.config.policy_fps), chunk_len - 1)
        self.steps_since_chunk += 1
        frame: dict[str, np.ndarray] = {key: value[0, frame_idx] for key, value in self.action_chunk.items()}

        action: torch.Tensor = torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))  # (num_envs, action_dim)
        # 양손 EE 목표 → pelvis 기준 wrist_yaw_link 목표 pos + quat xyzw
        for side, pos_idx, quat_idx in [
            ('left', LEFT_WRIST_POS_START_IDX, LEFT_WRIST_QUAT_START_IDX),
            ('right', RIGHT_WRIST_POS_START_IDX, RIGHT_WRIST_QUAT_START_IDX),
        ]:
            wrist_pos, wrist_quat = ee_rpy_to_wrist_pose(frame[f'action.{side}_ee_rpy'])
            action[:, pos_idx:pos_idx + 3] = torch.as_tensor(wrist_pos, dtype=action.dtype)
            action[:, quat_idx:quat_idx + 4] = torch.as_tensor(wrist_quat, dtype=action.dtype)
        # gripper 값 → hand_state
        action[:, LEFT_HAND_STATE_IDX] = gripper_value_to_hand_state(float(frame['action.left_gripper'][0]), self.config.unnorm_key)
        action[:, RIGHT_HAND_STATE_IDX] = gripper_value_to_hand_state(float(frame['action.right_gripper'][0]), self.config.unnorm_key)
        # base_command (vx, vy, vw, height) → navigate_cmd (3) + base_height_cmd (1). 다리 관절 출력(lower_body[:12])은 쓰지 않음
        action[:, NAVIGATE_CMD_START_IDX:NAVIGATE_CMD_START_IDX + 3] = torch.as_tensor(frame['action.base_command'][:3], dtype=action.dtype)
        action[:, BASE_HEIGHT_CMD_START_IDX] = float(frame['action.base_command'][3])
        # waist 관절 목표 (yaw, roll, pitch) → torso_orientation_rpy_cmd (roll, pitch, yaw)
        # waist 관절각을 torso 자세 명령으로 그대로 씀. 큰 허리 동작에서 차이가 나면 FK로 torso 자세를 계산
        waist_yaw, waist_roll, waist_pitch = frame['action.lower_body'][12:15]
        action[:, TORSO_ORIENTATION_RPY_CMD_START_IDX:TORSO_ORIENTATION_RPY_CMD_START_IDX + 3] = torch.tensor(
            [waist_roll, waist_pitch, waist_yaw], dtype=action.dtype
        )
        return action
