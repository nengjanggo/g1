'''
실제 unifolm-wla 모델에서 RTC sampling을 검증한다. unifolm-wla venv에서 g1 root 기준으로 실행:
    PYTHONPATH=third_party/unifolm-wla:. third_party/unifolm-wla/.venv/bin/python -m policy_server.check_rtc_unifolm \
        --ckpt_path checkpoints/UnifoLM-WLA-1.0-Base/checkpoints/model.safetensors

검증:
1. 같은 seed에서 max_guidance_weight=0 인 RTC sampling은 원본 predict_action과 같은 결과를 내야 한다.
2. 모델 자신의 다른 sample(seed 1)을 prev_action_chunk로 주면, RTC guided 결과(seed 0)의 frozen prefix가
   같은 seed의 일반 sampling보다 prev_action_chunk에 훨씬 가까워야 한다.
   (분포 밖 목표는 soft guidance로 완전히 따라가지 못하므로 쓰지 않는다)
'''
import argparse
import logging

import numpy as np
import torch

from model_server.action_server_wbc_msgpack_unitree import ActionServerWBCMsgpack
from policy_server.unifolm_rtc import enable_rtc


def main(
) -> None:
    '''dummy obs로 일반 / RTC sampling을 비교해 RTC가 prefix를 prev_action_chunk 쪽으로 끌어오는지 assert로 검증한다.'''
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--inference_delay', type=int, default=10)
    parser.add_argument('--prefix_attention_horizon', type=int, default=25)
    parser.add_argument('--max_guidance_weight', type=float, default=5.0)
    args: argparse.Namespace = parser.parse_args()

    # upstream main()과 같이 root logger handler를 교체 (rich handler는 upstream의 잘못된 logging 호출에서 예외를 던짐)
    logging.basicConfig(level=logging.WARNING, force=True)
    # 모델 로드 (upstream server의 obs → example 변환을 재사용)
    server: ActionServerWBCMsgpack = ActionServerWBCMsgpack(argparse.Namespace(
        ckpt_path=args.ckpt_path, instruction='pick up the cup', unnorm_key='UnifoLM_G1_Dex1',
        use_bf16=True, image_size=[320, 448], debug_save_dir=None,
    ))
    image: np.ndarray = np.zeros((480, 640, 3), dtype=np.uint8)  # (H_img, W_img, 3) BGR
    obs: dict = {
        'observation.images.cam_left_high': image,
        'observation.images.cam_left_wrist': image,
        'observation.images.cam_right_wrist': image,
        'observation.state.left_ee_6d': np.array([0.3, 0.2, 0.1, 1, 0, 0, 0, 1, 0], dtype=np.float32),   # (9,)
        'observation.state.right_ee_6d': np.array([0.3, -0.2, 0.1, 1, 0, 0, 0, 1, 0], dtype=np.float32),  # (9,)
        'observation.state.left_gripper': np.zeros(1, dtype=np.float32),   # (1,)
        'observation.state.right_gripper': np.zeros(1, dtype=np.float32),  # (1,)
        'observation.state.lower_body': np.zeros(15, dtype=np.float32),    # (15,)
    }
    example: dict = server._build_example(obs)['example']
    action_mask: torch.Tensor = torch.tensor(example['action_mask'])  # (D,)

    def sample(
        seed: int,
    ) -> torch.Tensor:
        '''seed를 고정해 server.model.predict_action으로 normalized action chunk (1, H, D)를 생성해 반환한다.'''
        torch.manual_seed(seed)
        with torch.no_grad():
            return torch.tensor(server.model.predict_action(examples=[example])['normalized_actions'])

    # 원본 sampling: prev_action_chunk(seed 1)와 비교 기준(seed 0)
    prev_action_chunk: torch.Tensor = sample(seed=1)  # (1, H, D)
    unguided: torch.Tensor = sample(seed=0)  # (1, H, D)

    def sample_rtc(
        max_guidance_weight: float,
    ) -> torch.Tensor:
        '''model.predict_action을 RTC 버전으로 교체한 뒤(submodule 파일은 수정하지 않음) seed 0으로 생성한 chunk (1, H, D)를 반환한다.'''
        enable_rtc(
            server.model,
            prev_action_chunk=prev_action_chunk.to(server.model.device),
            inference_delay=args.inference_delay,
            prefix_attention_horizon=args.prefix_attention_horizon,
            prefix_attention_schedule='exp',
            max_guidance_weight=max_guidance_weight,
        )
        return sample(seed=0)

    # 1. guidance 0이면 원본 sampling과 동일
    assert torch.allclose(sample_rtc(max_guidance_weight=0.0), unguided, atol=1e-3)

    guided: torch.Tensor = sample_rtc(max_guidance_weight=args.max_guidance_weight)  # (1, H, D)
    # 유한값, 무효 차원은 0 유지
    assert torch.isfinite(guided).all()
    assert (guided[..., ~action_mask] == 0).all()
    # 2. frozen prefix(앞 inference_delay step)의 prev_action_chunk 오차: guided가 unguided의 절반 미만
    prefix: slice = slice(0, args.inference_delay)
    error_guided: float = (guided[:, prefix] - prev_action_chunk[:, prefix]).norm().item()
    error_unguided: float = (unguided[:, prefix] - prev_action_chunk[:, prefix]).norm().item()
    assert error_guided < 0.5 * error_unguided, (error_guided, error_unguided)


if __name__ == '__main__':
    main()
