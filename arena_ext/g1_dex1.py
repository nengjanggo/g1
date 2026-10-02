'''
IsaacLab-Arena용 G1 + Dex1(2-finger gripper) embodiment.

Arena의 `g1_wbc_pink`(G1 + Dex3)를 상속해서 다음만 바꾼다.
- robot USD: unitree_sim_isaaclab의 G1-29dof + Dex1 wholebody USD (README의 asset 추출 참고)
- 손목 카메라 2개: unitree_sim_isaaclab의 Dex1 손목 카메라 설정
- 손 action: hand_state(0: open, 1: close)를 Dex1 prismatic 관절 목표 위치로 선형 변환

Arena venv에서 sim app이 뜬 뒤에 import되어야 하므로 `--policy_type arena_ext.g1_dex1.<Policy>`로 불러 등록한다.
Dex1 관련 값의 출처: https://github.com/unitreerobotics/unitree_sim_isaaclab (Apache-2.0)
  robots/unitree.py (G129_CFG_WITH_DEX1_WHOLEBODY), tasks/common_config/camera_configs.py,
'''

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import isaaclab.sim as sim_utils
import numpy as np
import torch
import warp as wp
from gymnasium.spaces.dict import Dict as GymSpacesDict
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationData
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

import arena_ext.plastic_box_parts_scene  # noqa: F401  장면 asset 등록
import isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action as pink_action_module
from arena_ext.unifolm_g1_convert import DEX1_CLOSE_POS, DEX1_OPEN_POS
from isaaclab_arena.assets.register import register_asset, register_policy
from isaaclab_arena.embodiments.g1.g1 import G1_CFG, G1CameraCfg, G1WBCPinkEmbodiment
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import G1DecoupledWBCPinkActionCfg
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.action_constants import (
    BASE_HEIGHT_CMD_START_IDX,
    LEFT_HAND_STATE_IDX,
    RIGHT_HAND_STATE_IDX,
)

G1_DEX1_USD_PATH: Path = (
    Path(__file__).resolve().parents[1] / 'assets/robots/g1-29dof_wholebody_dex1/g1_29dof_with_dex1_rev_1_0.usd'
)

# 손마다 두 손가락 관절. 순서: left finger 1, left finger 2, right finger 1, right finger 2
DEX1_JOINT_NAMES: list[str] = ['left_hand_Joint1_1', 'left_hand_Joint2_1', 'right_hand_Joint1_1', 'right_hand_Joint2_1']

# WBC joint order(43dof, Dex3 기준)에서 Dex1 관절이 차지할 Dex3 hand slot
DEX3_SLOT_TO_DEX1_JOINT: dict[str, str] = {
    'left_hand_index_0_joint': 'left_hand_Joint1_1',
    'left_hand_index_1_joint': 'left_hand_Joint2_1',
    'right_hand_index_0_joint': 'right_hand_Joint1_1',
    'right_hand_index_1_joint': 'right_hand_Joint2_1',
}


# robot articulation 설정: Arena G1 설정에서 USD, 초기 손 위치, 손 actuator만 교체
G1_DEX1_CFG: ArticulationCfg = G1_CFG.copy()
G1_DEX1_CFG.spawn.usd_path = str(G1_DEX1_USD_PATH)
G1_DEX1_CFG.init_state.joint_pos = {**G1_CFG.init_state.joint_pos, '.*_hand_Joint.*': DEX1_OPEN_POS}
# unitree의 friction=200은 Isaac Lab 3에서 의미가 달라질 수 있어 뺐다. gripper가 미끄러지면 추가 검토
G1_DEX1_CFG.actuators['hands'] = ImplicitActuatorCfg(
    joint_names_expr=['.*_hand_Joint.*'],
    stiffness=800.0,
    damping=3.0,
)


@configclass
class G1Dex1SceneCfg:
    robot: ArticulationCfg = G1_DEX1_CFG.copy()


def make_dex1_wrist_camera_cfg(
    side: str,
    pos_x_sign: float,
) -> CameraCfg:
    '''
    Dex1 손목 카메라 설정을 만든다.

    side('left' 또는 'right')의 hand_base_link에 unitree_sim_isaaclab과 같은 offset으로 카메라를 붙인
    CameraCfg를 반환한다. 좌우 카메라는 x offset 부호(pos_x_sign)만 다르다.
    '''
    # unitree 설정의 wxyz quaternion (-0.34202, 0.93969, 0, 0)을 Isaac Lab 3의 xyzw 순서로 바꿈
    offset: Pose = Pose(position_xyz=(pos_x_sign * 0.02541028, 0.045, 0.135), rotation_xyzw=(0.93969, 0.0, 0.0, -0.34202))
    return CameraCfg(
        prim_path=f'{{ENV_REGEX_NS}}/Robot/{side}_hand_base_link/{side}_wrist_cam',
        update_period=0.0,
        height=480,
        width=640,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=offset.position_xyz,
            rot=offset.rotation_xyzw,
            convention='ros',
        ),
    )


@configclass
class G1Dex1CameraCfg(G1CameraCfg):
    '''unitree_sim_isaaclab의 G1 머리 카메라(g1_front_camera) + Dex1 손목 카메라 2개.'''

    # Arena 머리 카메라(focal 15, 수평 화각 ~70°)는 실제 G1 데이터보다 좁아서 unitree 설정으로 교체:
    # d435_link에 붙여 link x축(47.6° 아래)을 보고, focal 7.6 / aperture 20 (수평 화각 ~105°)
    robot_head_cam: CameraCfg = CameraCfg(
        prim_path='{ENV_REGEX_NS}/Robot/d435_link/front_cam',
        update_period=0.0,
        height=480,
        width=640,
        data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=7.6,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
        ),
        # unitree 설정의 wxyz quaternion (0.5, -0.5, 0.5, -0.5)을 xyzw 순서로
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(-0.5, 0.5, -0.5, 0.5), convention='ros'),
    )
    left_wrist_cam: CameraCfg = make_dex1_wrist_camera_cfg('left', 1.0)
    right_wrist_cam: CameraCfg = make_dex1_wrist_camera_cfg('right', -1.0)


# WBC 출력 중 sim에 없는 Dex3 hand 관절은 조용히 건너뛰도록 교체 (원본은 관절마다 매 step print)
_postprocess_actions = pink_action_module.postprocess_actions


def postprocess_actions_sim_joints_only(
    wbc_action: dict,
    robot_data: ArticulationData,
    wbc_g1_joints_order: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    '''
    WBC 출력에서 sim articulation에 있는 관절만 골라 원본 postprocess_actions에 넘긴다.

    wbc_g1_joints_order 중 robot_data.joint_names에 없는 관절을 제외하고, sim 관절 순서의
    joint position target (num_envs, num_joints)를 반환한다.
    '''
    sim_joints_order: dict[str, int] = {
        name: index for name, index in wbc_g1_joints_order.items() if name in robot_data.joint_names
    }
    return _postprocess_actions(wbc_action, robot_data, sim_joints_order, device)


pink_action_module.postprocess_actions = postprocess_actions_sim_joints_only


class WbcPaddedJointData:
    '''
    WBC 관측용 robot data proxy.

    Arena WBC는 관절 수가 WBC joint order(43dof)와 같다고 가정한다. sim에 없는 Dex3 hand 관절을 0으로 채워
    joint_names / joint_pos / joint_vel / default_joint_pos를 43개로 맞추고, 나머지 속성은 원본 robot data를 그대로 쓴다.
    '''

    def __init__(
        self,
        robot_data: ArticulationData,
        wbc_g1_joints_order: dict[str, int],
    ):
        '''원본 robot data와 WBC joint order를 받아 관절 단위 값 4개를 0으로 padding해 둔다.'''
        self.robot_data: ArticulationData = robot_data
        missing_joint_names: list[str] = [name for name in wbc_g1_joints_order if name not in robot_data.joint_names]
        self.joint_names: list[str] = list(robot_data.joint_names) + missing_joint_names
        self.joint_pos: wp.array = self.pad_joint_values(robot_data.joint_pos, len(missing_joint_names))
        self.joint_vel: wp.array = self.pad_joint_values(robot_data.joint_vel, len(missing_joint_names))
        self.default_joint_pos: wp.array = self.pad_joint_values(robot_data.default_joint_pos, len(missing_joint_names))

    @staticmethod
    def pad_joint_values(
        joint_values: wp.array,
        num_missing_joints: int,
    ) -> wp.array:
        '''관절 값 (num_envs, num_sim_joints) 뒤에 0을 붙여 (num_envs, num_sim_joints + num_missing_joints)로 반환한다.'''
        joint_values_torch: torch.Tensor = wp.to_torch(joint_values)  # (num_envs, num_sim_joints)
        zeros: torch.Tensor = joint_values_torch.new_zeros(joint_values_torch.shape[0], num_missing_joints)  # (num_envs, num_missing_joints)
        return wp.from_torch(torch.cat([joint_values_torch, zeros], dim=1).contiguous())  # (num_envs, num_sim_joints + num_missing_joints)

    def __getattr__(
        self,
        name: str,
    ):
        '''padding하지 않은 속성(root pose, body state 등)은 원본 robot data에서 가져온다.'''
        return getattr(self.robot_data, name)


_prepare_observations = pink_action_module.prepare_observations


def prepare_observations_padded(
    num_envs: int,
    robot_data: ArticulationData,
    wbc_joints_order: dict[str, int],
) -> dict[str, np.ndarray]:
    '''sim에 없는 Dex3 hand 관절을 0으로 채운 robot data proxy로 원본 prepare_observations를 호출해 WBC 관측을 반환한다.'''
    return _prepare_observations(num_envs, WbcPaddedJointData(robot_data, wbc_joints_order), wbc_joints_order)


pink_action_module.prepare_observations = prepare_observations_padded


class G1Dex1WBCPinkAction(G1DecoupledWBCPinkAction):
    '''G1 WBC + PINK IK action에 Dex1 gripper 제어를 더한 action term.'''

    def __init__(
        self,
        cfg: G1DecoupledWBCPinkActionCfg,
        env: ManagerBasedEnv,
    ):
        '''
        원본 action term을 만든 뒤, WBC joint order의 Dex3 hand slot 일부를 Dex1 관절 이름으로 바꾸고
        Dex1 관절의 sim index를 찾아 둔다.
        '''
        super().__init__(cfg, env)
        # WBC 관측 변환이 모든 sim 관절을 요구하므로 Dex1 관절을 Dex3 hand slot에 배정
        for dex3_slot, dex1_joint in DEX3_SLOT_TO_DEX1_JOINT.items():
            self.wbc_g1_joints_order[dex1_joint] = self.wbc_g1_joints_order.pop(dex3_slot)
        self._dex1_joint_ids: list[int] = self._asset.find_joints(DEX1_JOINT_NAMES, preserve_order=True)[0]

    def process_actions(
        self,
        actions: torch.Tensor,
    ) -> None:
        '''
        원본 WBC + IK로 몸 관절 목표를 계산한 뒤, hand_state를 Dex1 관절 목표 위치로 바꿔 덮어쓴다.

        actions: (num_envs, action_dim). hand_state 0이면 open, 1이면 close, 그 사이는 선형 보간.
        '''
        super().process_actions(actions)
        # hand_state를 [0, 1]로 자르고 open/close 위치 사이로 선형 보간
        hand_state: torch.Tensor = actions[:, [LEFT_HAND_STATE_IDX, RIGHT_HAND_STATE_IDX]].clamp(0.0, 1.0)  # (num_envs, 2)
        gripper_pos: torch.Tensor = DEX1_OPEN_POS + hand_state * (DEX1_CLOSE_POS - DEX1_OPEN_POS)  # (num_envs, 2)
        # 손마다 두 손가락 관절에 같은 목표를 넣음
        self._processed_actions[:, self._dex1_joint_ids] = gripper_pos.repeat_interleave(2, dim=1)  # (num_envs, 2) -> (num_envs, 4)


@register_asset
class G1Dex1WBCPinkEmbodiment(G1WBCPinkEmbodiment):
    '''G1 + Dex1 gripper + 손목 카메라 2개, WBC + PINK IK 제어 embodiment.'''

    name = 'g1_dex1_wbc_pink'
    tags = ['embodiment']

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        lock_waist: bool = False,
    ):
        '''Arena G1 WBC + PINK embodiment를 만든 뒤 robot, 카메라, action term class만 Dex1용으로 교체한다.'''
        super().__init__(enable_cameras, initial_pose, lock_waist)
        self.scene_config = G1Dex1SceneCfg()
        self.camera_config = G1Dex1CameraCfg()
        self.action_config.g1_action.class_type = G1Dex1WBCPinkAction


@dataclass
class GripperToggleCheckPolicyCfg(PolicyCfg):
    '''Dex1 gripper 매핑 확인용 policy 설정.'''

    toggle_period_steps: int = 100

    # 목표 골반 높이 [m]. 0이면 WBC가 바닥까지 주저앉으려다 넘어진다 (Arena G1 test도 0.75 사용)
    base_height_cmd: float = 0.75


@register_policy
class GripperToggleCheckPolicy(PolicyBase[GripperToggleCheckPolicyCfg]):
    '''팔/이동 명령은 0, 골반 높이는 base_height_cmd로 두고 양손 hand_state만 일정 주기로 열고 닫는 확인용 policy.'''

    name = 'g1_dex1_gripper_toggle_check'

    def __init__(
        self,
        config: GripperToggleCheckPolicyCfg,
    ):
        '''step counter를 0으로 초기화한다.'''
        super().__init__(config)
        self.step_count: int = 0

    def get_action(
        self,
        env: gym.Env,
        observation: GymSpacesDict,
    ) -> torch.Tensor:
        '''
        0 action에 base_height_cmd를 넣고, hand_state는 toggle_period_steps마다 0(open)과 1(close)을 번갈아 넣어 반환한다.

        반환 shape: (num_envs, action_dim)
        '''
        action: torch.Tensor = torch.zeros(env.action_space.shape, device=torch.device(env.unwrapped.device))  # (num_envs, action_dim)
        action[:, BASE_HEIGHT_CMD_START_IDX] = self.config.base_height_cmd
        hand_state: float = float((self.step_count // self.config.toggle_period_steps) % 2)
        action[:, [LEFT_HAND_STATE_IDX, RIGHT_HAND_STATE_IDX]] = hand_state
        self.step_count += 1
        return action
