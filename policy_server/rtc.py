# Real-Time Chunking (RTC) guided flow sampling.
#
# Ported from Physical Intelligence's official implementation:
#   https://github.com/Physical-Intelligence/real-time-chunking-kinetix
#   src/model.py @ 9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b
#   (get_prefix_weights, FlowPolicy.realtime_action)
#   MIT License, Copyright (c) 2025 Physical Intelligence
# The original is JAX. Only the JAX -> PyTorch translation was done; the algorithm and constants match the original.
# Not ported: realtime_action's `simulated_delay` branch (applies only to training-time delay simulation).
from typing import Callable, Literal

import torch

PrefixAttentionSchedule = Literal['linear', 'exp', 'ones', 'zeros']


def get_prefix_weights(
    start: int,
    end: int,
    total: int,
    schedule: PrefixAttentionSchedule,
) -> torch.Tensor:
    '''
    action chunk의 각 timestep이 이전 chunk(prefix)를 얼마나 따를지 나타내는 weight (total,)를 반환한다.
    start=2, end=6, total=10, schedule='linear' 이면 [1, 1, 4/5, 3/5, 2/5, 1/5, 0, 0, 0, 0].
    start(포함)부터 chunk가 바뀔 수 있고, end(미포함)부터 prefix를 무시한다. end < start 이면 start를 end로 내린다.
    '''
    # end가 start보다 우선
    start = min(start, end)
    index: torch.Tensor = torch.arange(total, dtype=torch.float32)  # (total,)
    if schedule == 'ones':
        w: torch.Tensor = torch.ones(total)
    elif schedule == 'zeros':
        w = (index < start).float()
    elif schedule == 'linear' or schedule == 'exp':
        w = torch.clip((start - 1 - index) / (end - start + 1) + 1, 0, 1)
        if schedule == 'exp':
            w = w * torch.expm1(w) / (torch.e - 1)
    else:
        raise ValueError(f'Invalid schedule: {schedule}')
    return torch.where(index >= end, 0.0, w)


def guided_velocity(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    x_t: torch.Tensor,
    t: float,
    prev_action_chunk: torch.Tensor,
    weights: torch.Tensor,
    max_guidance_weight: float,
) -> torch.Tensor:
    '''
    한 denoising step에서 velocity_fn(x_t, t)의 velocity v_t에 prefix inpainting용 pseudo-inverse guidance를 더해 반환한다.
    원본 realtime_action 내부의 pinv_corrected_velocity에 해당한다.

    x_t, prev_action_chunk, 반환값: (B, H, D). B=batch, H=action chunk 길이, D=action 차원
    weights: (H,) get_prefix_weights 결과
    t: flow time (0=noise, 1=data)
    '''
    with torch.enable_grad():
        # denoiser: x_t에서 한 번에 예측한 clean action x_1 = x_t + v_t * (1 - t)
        x_t = x_t.detach().requires_grad_(True)
        v_t: torch.Tensor = velocity_fn(x_t, t)  # (B, H, D)
        x_1: torch.Tensor = x_t + v_t * (1 - t)  # (B, H, D)
        # prefix 오차를 cotangent로 한 VJP: (∂x_1/∂x_t)^T · error
        error: torch.Tensor = ((prev_action_chunk - x_1) * weights.to(x_1)[None, :, None]).detach()  # (B, H, D)
        pinv_correction: torch.Tensor = torch.autograd.grad(x_1, x_t, grad_outputs=error)[0]  # (B, H, D)
    # 논문의 guidance weight 상수 (t=0 이면 (1-t)/t = inf → max_guidance_weight)
    inv_r2: float = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
    c: float = max_guidance_weight if t == 0 else (1 - t) / t
    guidance_weight: float = min(c * inv_r2, max_guidance_weight)
    return v_t.detach() + guidance_weight * pinv_correction


def realtime_action(
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor],
    noise: torch.Tensor,
    num_steps: int,
    prev_action_chunk: torch.Tensor,
    inference_delay: int,
    prefix_attention_horizon: int,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float,
) -> torch.Tensor:
    '''
    RTC guided Euler sampling으로 action chunk x_1을 생성해 반환한다.
    앞쪽 inference_delay step은 prev_action_chunk를 그대로 따르고, prefix_attention_horizon까지 점차 자유로워진다.

    velocity_fn(x_t, t) -> v_t: (B, H, D) flow velocity
    noise, prev_action_chunk, 반환값: (B, H, D)
    '''
    dt: float = 1 / num_steps
    weights: torch.Tensor = get_prefix_weights(
        inference_delay, prefix_attention_horizon, noise.shape[1], prefix_attention_schedule,
    )  # (H,)
    x_t: torch.Tensor = noise
    for step in range(num_steps):
        t: float = step * dt
        v_t: torch.Tensor = guided_velocity(velocity_fn, x_t, t, prev_action_chunk, weights, max_guidance_weight)
        x_t = x_t + dt * v_t
    return x_t


if __name__ == '__main__':
    # get_prefix_weights docstring 예시 검증
    expected: torch.Tensor = torch.tensor([1, 1, 4 / 5, 3 / 5, 2 / 5, 1 / 5, 0, 0, 0, 0])
    assert torch.allclose(get_prefix_weights(2, 6, 10, 'linear'), expected)

    # toy flow: x_1이 x_t에 의존하도록(∂x_1/∂x_t ≠ 0) 만든 velocity. 의존이 없으면 guidance가 x_1을 바꿀 수 없음
    torch.manual_seed(0)
    B, H, D = 2, 10, 3
    target: torch.Tensor = torch.randn(B, H, D)  # (B, H, D)
    linear_layer: torch.nn.Linear = torch.nn.Linear(D, D)

    def toy_velocity_fn(
        x_t: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        '''target 방향 항과 x_t 의존 항(linear_layer)을 더한 toy velocity (B, H, D)를 반환한다.'''
        return target + 0.5 * linear_layer(x_t)

    noise: torch.Tensor = torch.randn(B, H, D)
    prev_action_chunk: torch.Tensor = torch.randn(B, H, D)
    common: dict = dict(num_steps=4, prev_action_chunk=prev_action_chunk, inference_delay=3,
                        prefix_attention_horizon=7, prefix_attention_schedule='exp')
    with torch.no_grad():
        unguided: torch.Tensor = realtime_action(toy_velocity_fn, noise, max_guidance_weight=0.0, **common)
        guided: torch.Tensor = realtime_action(toy_velocity_fn, noise, max_guidance_weight=5.0, **common)

    # max_guidance_weight=0 이면 일반 Euler sampling과 동일
    x_t: torch.Tensor = noise
    for step in range(4):
        x_t = x_t + 0.25 * toy_velocity_fn(x_t, step * 0.25)
    assert torch.allclose(unguided, x_t, atol=1e-5)

    # guidance가 있으면 frozen prefix(앞 inference_delay step)가 prev_action_chunk에 더 가까워짐
    prefix_error_unguided: float = (unguided[:, :3] - prev_action_chunk[:, :3]).norm().item()
    prefix_error_guided: float = (guided[:, :3] - prev_action_chunk[:, :3]).norm().item()
    assert prefix_error_guided < 0.5 * prefix_error_unguided, (prefix_error_guided, prefix_error_unguided)
