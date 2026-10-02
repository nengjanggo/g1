'''
unifolm-wla Unitree msgpack server 실행 진입점. upstream server에 unnorm_key별 action mask만 바꿔 끼운다.

upstream server는 항상 Dex1 기준 action mask(base pose [35:41] 꺼짐)를 쓴다. WBT 데이터는 학습 때 base pose도 켜져 있었으므로
`UnifoLM_WBT` 요청에는 base pose를 켠 mask를 넘긴다. 이 mask에서 WBT dataset의 걷는 구간 예측 vx가 0에서 실제 방향으로 바뀌는 것을 확인했다.

사용법 (third_party/unifolm-wla에서, upstream server와 같은 인자):
    PYTHONPATH=.:../.. .venv/bin/python -m policy_server.unitree_server --ckpt_path ... --host 127.0.0.1 --port 8600
'''

import numpy as np

import model_server.action_server_wbc_msgpack_unitree as upstream
from model_server.unifolm_wla_action_adapter import build_unitree_fullbody_action_mask
from unifolm_wla.dataloader.multi_source_dataset.action_mapping import SLICES

# WBT 요청용 action mask: upstream mask + base pose [35:41].
# 학습 설정상 WBT의 gripper 자리는 꺼져 있었을 수 있지만, sim에서 gripper를 제어하려고 켜 둔다
WBT_ACTION_MASK: np.ndarray = build_unitree_fullbody_action_mask()  # (54,)
WBT_ACTION_MASK[SLICES['base_rotvec']] = True


class ActionServerWithKeyMask(upstream.ActionServerWBCMsgpack):
    '''upstream server에서 unnorm_key가 UnifoLM_WBT인 요청만 WBT_ACTION_MASK를 쓰도록 바꾼 server.'''

    def _build_example(
        self,
        obs: dict,
    ) -> dict:
        '''upstream 전처리 결과에서 WBT 요청의 action mask만 WBT_ACTION_MASK로 교체해 반환한다.'''
        prep: dict = super()._build_example(obs)
        if prep['unnorm_key'] == 'UnifoLM_WBT':
            prep['example']['action_mask'] = WBT_ACTION_MASK
        return prep


if __name__ == '__main__':
    # upstream main()이 생성하는 server class를 교체한 뒤 그대로 실행
    upstream.ActionServerWBCMsgpack = ActionServerWithKeyMask
    upstream.main()
