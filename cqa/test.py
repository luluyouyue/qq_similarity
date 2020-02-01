# encoding=utf8
import numpy as np
from . import qa


def to_unicode(o):
    print('type o = ', type(o))
    if isinstance(o, str):
        return o
    return o


input_question =  ['<s>', '哺乳期', '能', '用', '碧欧泉', '的', '护肤品', '吗', '？', '</s>', '<pad>', '<pad>', '<pad>', '<pad>',
'<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>',
'<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>']


vocab_id = {}
def to_padded_token_ids(tokenizd_input):
    _tokenizd_input = np.asarray(tokenizd_input)
    raw_shape = _tokenizd_input.shape
    _tokenizd_input = np.reshape(_tokenizd_input, [-1, raw_shape[-1]])
    if raw_shape[-1] < 40:
        _tokenizd_input = np.concatenate([_tokenizd_input,
                                          np.full([reduce(lambda x, y: x * y, raw_shape[:-1]),
                                                   40 - raw_shape[-1]], PAD)], axis=-1)
    else:
        _tokenizd_input = _tokenizd_input[:, :40]
    padded_shape = [i for i in raw_shape[:-1]]
    padded_shape.append(40)
    return np.reshape(
        [vocab_id.get(to_unicode(x).strip(), 'UNK_ID') for x in np.reshape(_tokenizd_input, [-1])],
        padded_shape)

ided_question = to_padded_token_ids(input_question)
