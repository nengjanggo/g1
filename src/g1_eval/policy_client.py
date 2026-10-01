import functools
from typing import Any

import msgpack
import numpy as np
from websockets.sync.client import ClientConnection, connect


def _pack_array(
    obj: Any,
) -> Any:
    '''
    np.ndarray / np.generic을 unifolm-wla model_server의 msgpack_numpy 포맷 dict로 변환해 반환한다.
    그 외 객체는 그대로 반환한다.
    '''
    # ndarray를 bytes + dtype + shape로 직렬화
    if isinstance(obj, np.ndarray):
        return {b'__ndarray__': True, b'data': obj.tobytes(), b'dtype': obj.dtype.str, b'shape': obj.shape}
    # numpy scalar를 python scalar + dtype로 직렬화
    if isinstance(obj, np.generic):
        return {b'__npgeneric__': True, b'data': obj.item(), b'dtype': obj.dtype.str}
    return obj


def _unpack_array(
    obj: dict,
) -> Any:
    '''
    _pack_array로 직렬화된 dict를 np.ndarray / np.generic으로 복원해 반환한다.
    해당 포맷이 아닌 dict는 그대로 반환한다.
    '''
    if b'__ndarray__' in obj:
        return np.ndarray(buffer=obj[b'data'], dtype=np.dtype(obj[b'dtype']), shape=obj[b'shape'])
    if b'__npgeneric__' in obj:
        return np.dtype(obj[b'dtype']).type(obj[b'data'])
    return obj


_packb = functools.partial(msgpack.packb, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class PolicyClient:
    '''unifolm-wla model_server(websocket + msgpack)에 obs를 보내고 action chunk를 받는 client.'''

    def __init__(
        self,
        host: str,
        port: int,
    ) -> None:
        '''
        server에 연결하고, 연결 직후 server가 보내는 metadata(dict)를 self.metadata에 저장한다.
        '''
        # server에 연결
        self.ws: ClientConnection = connect(f'ws://{host}:{port}', max_size=None, compression=None)
        # 연결 직후 전송되는 metadata 수신 (data_keys, action_chunk_size 등)
        self.metadata: dict = _unpackb(self.ws.recv())

    def get_action(
        self,
        obs: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        '''
        obs dict(metadata['data_keys']의 observation.* key)를 보내고,
        action dict(action.* key → (1, T, D) array, unnormalized absolute)를 반환한다.
        '''
        # get_action 요청 전송
        self.ws.send(_packb({'type': 'get_action', 'obs': obs}))
        reply: bytes | str = self.ws.recv()
        # server 예외 발생 시 traceback이 str로 전송됨
        if isinstance(reply, str):
            raise RuntimeError(reply)
        return _unpackb(reply)

    def close(
        self,
    ) -> None:
        '''server 연결을 종료한다.'''
        self.ws.close()
