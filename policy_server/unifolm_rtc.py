import functools
from typing import Callable

import torch

from policy_server.rtc import PrefixAttentionSchedule, realtime_action


def make_velocity_fn(
    action_head: torch.nn.Module,
    vl_embs: torch.Tensor,
    state: torch.Tensor | None,
    action_mask: torch.Tensor | None,
    encoder_attention_mask: torch.Tensor | None,
    body_type_ids: torch.Tensor | None,
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    '''
    unifolm-wla MMDiT action head의 denoising 한 step(MMDiT_ActionHeader.predict_action loop body)을
    velocity_fn(x_t, t) -> v_t 로 감싸 반환한다. condition(vl_embs, state 등)은 closure로 고정한다.

    vl_embs: (B, L, C) VLM last hidden. L=token 수, C=hidden 차원
    action_mask: (B, D) 유효 action 차원 mask. D=54
    x_t, v_t: (B, H, D). H=action_horizon
    '''
    batch_size: int = vl_embs.shape[0]
    device: torch.device = vl_embs.device
    # 유효 차원 mask를 horizon 방향으로 확장: (B, D) → (B, H, D)
    mask_part: torch.Tensor | None = (
        action_mask.float().unsqueeze(1).expand(-1, action_head.action_horizon, -1).contiguous()
        if action_mask is not None else None
    )
    # state feature는 step과 무관하므로 한 번만 계산
    state_features: torch.Tensor | None = (
        action_head.state_encoder(state) if (action_head.state_encoder is not None and state is not None) else None
    )

    def velocity_fn(
        x_t: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        '''x_t (B, H, D)와 flow time t에서 action head가 예측한 velocity (B, H, D)를 반환한다.'''
        # 연속 time → action head의 discrete timestep bucket
        t_discretized: int = int(t * action_head.num_timestep_buckets)
        timesteps_tensor: torch.Tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)  # (B,)

        # 무효 차원을 입력 전에 0으로 만들어, RTC VJP correction이 무효 차원으로 새지 않게 함
        if mask_part is not None:
            encoder_input: torch.Tensor = torch.cat([x_t * mask_part, mask_part], dim=2)  # (B, H, 2D)
        else:
            encoder_input = x_t  # (B, H, D)
        action_features: torch.Tensor = action_head.action_encoder(encoder_input, timesteps_tensor)
        action_features = action_head._add_action_pos_embed(action_features, device)
        action_features = action_head._add_embodiment_embed(action_features, body_type_ids)
        hidden_states: torch.Tensor = (
            torch.cat((state_features, action_features), dim=1) if state_features is not None else action_features
        )

        model_output: torch.Tensor = action_head.model(
            hidden_states=hidden_states,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timesteps_tensor,
        )
        model_output = action_head._add_embodiment_embed_dec(model_output, body_type_ids)
        # 뒤쪽 H개 token이 action velocity
        pred_velocity: torch.Tensor = action_head.action_decoder(model_output)[:, -action_head.action_horizon:]  # (B, H, D)
        return pred_velocity * mask_part if mask_part is not None else pred_velocity

    return velocity_fn


def rtc_predict_action(
    action_head: torch.nn.Module,
    vl_embs: torch.Tensor,
    state: torch.Tensor | None = None,
    action_mask: torch.Tensor | None = None,
    encoder_attention_mask: torch.Tensor | None = None,
    body_type_ids: torch.Tensor | None = None,
    *,
    prev_action_chunk: torch.Tensor,
    inference_delay: int,
    prefix_attention_horizon: int,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float,
) -> torch.Tensor:
    '''
    MMDiT_ActionHeader.predict_action과 같은 signature/출력의 RTC 버전.
    functools.partial로 RTC 인자를 고정해 model.action_model.predict_action을 instance 단위로 덮어써서 사용한다.

    prev_action_chunk: (B, H, D) 이전 chunk를 현재 state 기준으로 다시 표현한 normalized unified action
    반환값: (B, H, D) normalized unified action chunk
    '''
    velocity_fn: Callable[[torch.Tensor, float], torch.Tensor] = make_velocity_fn(
        action_head, vl_embs, state, action_mask, encoder_attention_mask, body_type_ids,
    )
    # 원본과 같은 초기 noise (무효 차원은 0)
    noise: torch.Tensor = torch.randn(
        size=(vl_embs.shape[0], action_head.action_horizon, action_head.action_output_dim),
        dtype=vl_embs.dtype,
        device=vl_embs.device,
    )  # (B, H, D)
    if action_mask is not None:
        noise = noise * action_mask.to(noise.dtype).unsqueeze(1)
    return realtime_action(
        velocity_fn,
        noise,
        num_steps=action_head.num_inference_timesteps,
        prev_action_chunk=prev_action_chunk.to(noise),
        inference_delay=inference_delay,
        prefix_attention_horizon=prefix_attention_horizon,
        prefix_attention_schedule=prefix_attention_schedule,
        max_guidance_weight=max_guidance_weight,
    )


def enable_rtc(
    model: torch.nn.Module,
    prev_action_chunk: torch.Tensor,
    inference_delay: int,
    prefix_attention_horizon: int,
    prefix_attention_schedule: PrefixAttentionSchedule,
    max_guidance_weight: float,
) -> None:
    '''
    unifolm-wla framework model의 predict_action이 RTC sampling을 쓰도록 instance 단위로 교체한다 (submodule 파일은 수정하지 않음).
    prev_action_chunk가 매 요청마다 바뀌므로 매 추론 전에 다시 호출한다.

    prev_action_chunk: (B, H, D) 이전 chunk를 현재 state 기준으로 다시 표현한 normalized unified action
    '''
    # framework predict_action의 @torch.inference_mode를 no_grad로 대체 (inference mode 안에서는 enable_grad로도 VJP 불가)
    model.predict_action = functools.partial(torch.no_grad()(type(model).predict_action.__wrapped__), model)
    # action head의 sampling loop를 RTC 버전으로 교체
    model.action_model.predict_action = functools.partial(
        rtc_predict_action,
        model.action_model,
        prev_action_chunk=prev_action_chunk,
        inference_delay=inference_delay,
        prefix_attention_horizon=prefix_attention_horizon,
        prefix_attention_schedule=prefix_attention_schedule,
        max_guidance_weight=max_guidance_weight,
    )
